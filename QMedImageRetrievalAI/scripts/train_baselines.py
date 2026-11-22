#!/usr/bin/env python3
"""Train ResNet/DenseNet/ViT/optional CLIP, BiomedCLIP, DINOv2 with same retrieval loss.

For pretrained optional foundation models: compatible packages and weights are needed.
Untrained weights are a software sanity check, NOT fair published baselines.
"""
import argparse
import sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pandas as pd
from qmed.data.dataset import MedicalDataset
from qmed.evaluation.metrics import evaluate_embeddings
from qmed.retrieval.embedding import extract_embeddings
from qmed.training.trainer import train_model,load_checkpoint
from qmed.utils.config import load_config,resolve_device,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/default.yaml")
    p.add_argument("--manifest")
    p.add_argument("--models",default="resnet50,densenet121,vit_b_16")
    p.add_argument("--pretrained",action="store_true",help="Load external pretrained weights (requires network or cache)")
    p.add_argument("--openclip-model",help="e.g. ViT-B-32 or hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    p.add_argument("--openclip-pretrained",help="e.g. openai")
    p.add_argument("--hf-vision-id",help="Hugging Face vision repository id")
    p.add_argument("--out",default="outputs/baselines")
    p.add_argument("--seed",type=int,default=0)
    args=p.parse_args()
    base=load_config(args.config)
    if args.manifest: base["data"]["manifest"]=args.manifest
    rows=[]
    for name in [x.strip() for x in args.models.split(",") if x.strip()]:
        cfg=deepcopy(base)
        cfg["seed"]=args.seed
        cfg["model"].update({"kind":"baseline","baseline_backbone":name,"pretrained":args.pretrained})
        if args.openclip_model: cfg["model"]["openclip_model"]=args.openclip_model
        if args.openclip_pretrained: cfg["model"]["openclip_pretrained"]=args.openclip_pretrained
        if args.hf_vision_id: cfg["model"]["hf_vision_id"]=args.hf_vision_id
        root=Path(args.out)/name
        checkpoint=train_model(cfg,root)
        model,cfg,_=load_checkpoint(checkpoint,resolve_device(cfg["training"].get("device","auto")))
        ds=MedicalDataset(cfg["data"]["manifest"],image_size=cfg["data"]["image_size"],split='test',
                          quality_filter=cfg["data"].get("quality_filter",False),
                          min_quality=cfg["data"].get("min_quality",.35))
        vec,meta=extract_embeddings(model,ds,next(model.parameters()).device,cfg['data']['batch_size'],cfg['data']['workers'])
        metrics,per=evaluate_embeddings(vec,meta,k_values=cfg['retrieval']['k_values'])
        per.to_csv(root/'test_per_query.csv',index=False)
        write_json(root/'test_metrics.json',metrics)
        rows.append({"model":name,"pretrained":args.pretrained,**metrics})
        pd.DataFrame(rows).to_csv(Path(args.out)/'comparison.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    main()
