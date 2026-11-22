#!/usr/bin/env python3
"""Train and evaluate real ablations over matched seeds; NEVER fabricate metrics."""
from __future__ import annotations
import argparse
import sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
from qmed.data.dataset import MedicalDataset
from qmed.evaluation.metrics import evaluate_embeddings
from qmed.evaluation.statistics import paired_seed_statistics,holm_adjust
from qmed.retrieval.embedding import extract_embeddings
from qmed.training.trainer import load_checkpoint,train_model
from qmed.utils.config import load_config,resolve_device,write_json

VARIANTS=("full","no_quantum","classical_control","no_alignment","no_mri_attention",
          "no_contrastive","no_triplet","no_regularization","no_crossmodal_alignment")


def apply_variant(config,variant):
    cfg=deepcopy(config)
    if variant == "no_quantum":
        cfg["model"]["quantum_backend"]="none"
    elif variant == "classical_control":
        cfg["model"]["quantum_backend"]="classical_control"
    elif variant == "no_alignment":
        cfg["model"]["use_alignment"]=False
    elif variant == "no_mri_attention":
        cfg["model"]["use_mri_attention"]=False
    elif variant == "no_contrastive":
        cfg["training"]["lambda_contrastive"]=0.0
    elif variant == "no_triplet":
        cfg["training"]["lambda_triplet"]=0.0
    elif variant == "no_regularization":
        cfg["training"]["lambda_regularization"]=0.0
    elif variant == "no_crossmodal_alignment":
        cfg["training"]["lambda_alignment"]=0.0
    elif variant != "full":
        raise ValueError(f"unknown variant={variant}")
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/default.yaml")
    p.add_argument("--manifest")
    p.add_argument("--variants",default="full,no_quantum,classical_control,no_alignment,no_mri_attention,no_contrastive,no_triplet")
    p.add_argument("--seeds",default="0,1,2,3,4")
    p.add_argument("--quantum-grid",action="store_true",help="Also sweep widths 4,6,8,10 and layers 2,4,6,8")
    p.add_argument("--out",default="outputs/ablations")
    p.add_argument("--force",action="store_true",help="Retrain even if checkpoint already exists")
    args=p.parse_args()
    base=load_config(args.config)
    if args.manifest:
        base["data"]["manifest"]=args.manifest
    seeds=[int(s) for s in args.seeds.split(",") if s.strip()]
    variants=[s.strip() for s in args.variants.split(",") if s.strip()]
    extras={}
    if args.quantum_grid:
        for q in (4,6,8,10):
            for layer in (2,4,6,8):
                extras[f"qubits{q}_layers{layer}"]=(q,layer)
        variants.extend(extras)
    if any(v not in VARIANTS and v not in extras for v in variants):
        p.error("invalid --variants")
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    scores=[]
    for variant in variants:
        for seed in seeds:
            cfg=apply_variant(base,"full") if variant in extras else apply_variant(base,variant)
            if variant in extras:
                q,layer=extras[variant]
                cfg["model"]["qubits"]=q
                cfg["model"]["quantum_layers"]=layer
                cfg["model"]["projected_dim"]=2*q*layer
            cfg["seed"]=seed
            folder=root/variant/f"seed_{seed}"
            best=folder/"best.pt"
            if args.force or not best.is_file():
                best=train_model(cfg,folder,verbose=True)
            device=resolve_device(cfg["training"].get("device","auto"))
            model,cfg,_=load_checkpoint(best,device)
            test=MedicalDataset(cfg["data"]["manifest"],image_size=cfg["data"]["image_size"],split="test",
                                roi_policy=cfg["data"].get("roi_policy","none"),
                                quality_filter=cfg["data"].get("quality_filter",False),
                                min_quality=cfg["data"].get("min_quality",.35))
            embeddings,metadata=extract_embeddings(model,test,device,cfg["data"]["batch_size"],cfg["data"].get("workers",0))
            result,per=evaluate_embeddings(embeddings,metadata,k_values=cfg["retrieval"].get("k_values",[5,10,20]))
            per.to_csv(folder/"test_per_query.csv",index=False)
            write_json(folder/"test_metrics.json",result)
            scores.append({"variant":variant,"seed":seed,**result})
            pd.DataFrame(scores).to_csv(root/"scores_by_seed.csv",index=False)
    frame=pd.DataFrame(scores)
    summary=frame.groupby("variant",as_index=False).agg(mAP_mean=("mAP","mean"),mAP_sd=("mAP","std"),
                                                       MRR_mean=("MRR","mean"),n_runs=("seed","count"))
    summary.to_csv(root/"ablation_summary.csv",index=False)
    comparisons=[]
    if "full" in frame.variant.values:
        control=frame[frame.variant=="full"][['seed','mAP']].rename(columns={'mAP':'full_mAP'})
        for variant in variants:
            if variant=="full": continue
            ab=frame[frame.variant==variant][['seed','mAP']]
            matched=control.merge(ab,on='seed').dropna()
            if len(matched)>=2:
                try:
                    comparisons.append({"variant":variant,**paired_seed_statistics(matched.full_mAP.to_list(),matched.mAP.to_list())})
                except ValueError:
                    pass
    if comparisons:
        for col in ('t_pvalue','wilcoxon_pvalue'):
            adjusted=holm_adjust([c[col] for c in comparisons])
            for c,a in zip(comparisons,adjusted):
                c[col+'_holm']=a
        pd.DataFrame(comparisons).to_csv(root/'paired_significance.csv',index=False)
    print(summary.to_string(index=False))
    print(f"Saved actual experiment results to {root.resolve()}")

if __name__=="__main__":
    main()
