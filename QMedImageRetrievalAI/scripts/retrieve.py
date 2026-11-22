#!/usr/bin/env python3
"""Search the medical repository with a new CT/MRI/X-ray query, no query labels needed."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
from qmed.training.trainer import load_checkpoint
from qmed.data.dataset import MODALITY_TO_ID
from qmed.data.preprocessing import read_grayscale,normalize_image
from qmed.retrieval.faiss_index import RetrievalIndex
from qmed.utils.config import resolve_device,write_json

def embed_query(model, device, image_size, modality, image=None, sequences=None, slice_idx=0):
    sequences=sequences or []
    if modality == "MRI" and sequences:
        if len(sequences)!=4:
            raise ValueError("Provide ALL four MRI sequences in T1,T1ce,T2,FLAIR order")
        tensors=[normalize_image(read_grayscale(p,slice_idx),image_size) for p in sequences]
    else:
        if not image:
            raise ValueError("--image is required except when --t1/--t1ce/--t2/--flair are given")
        mono=normalize_image(read_grayscale(image,slice_idx),image_size)
        tensors=[mono]+[torch.zeros_like(mono)]*3
    stacked=torch.cat(tensors,dim=0)[None].to(device)
    with torch.inference_mode():
        z=model(stacked,torch.tensor([MODALITY_TO_ID[modality]],device=device))
    return z[0].cpu().numpy()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--index-dir",required=True)
    p.add_argument("--image",help="CT/X-ray image; MRI single-sequence fallback")
    p.add_argument("--modality",choices=["CT","MRI","XRAY"],required=True)
    for x in ("t1","t1ce","t2","flair"):
        p.add_argument("--"+x)
    p.add_argument("--slice",type=int,default=0,help="slice index for NIfTI/DICOM-3D")
    p.add_argument("--target",choices=["CT","MRI","XRAY"])
    p.add_argument("--exclude-image-id")
    p.add_argument("--exclude-patient-id")
    p.add_argument("--k",type=int,default=10)
    p.add_argument("--label-aware-category",type=int,choices=range(5),help="Optional explicit 0..4 category (NOT primary unlabeled evaluation)")
    p.add_argument("--out",default="outputs/query_results.json")
    p.add_argument("--device",default="auto")
    args=p.parse_args()
    device=resolve_device(args.device)
    model,cfg,_=load_checkpoint(args.checkpoint,device)
    seqs=[getattr(args,s) for s in ("t1","t1ce","t2","flair")]
    if any(seqs) and not all(seqs):
        p.error("For MRI provide all --t1/--t1ce/--t2/--flair or use --image alone")
    query=embed_query(model,device,cfg["data"]["image_size"],args.modality,args.image,
                      sequences=seqs if all(seqs) else None,slice_idx=args.slice)
    idx=RetrievalIndex.load(args.index_dir)
    c=cfg["retrieval"]
    results=idx.search(query,k=args.k,candidate_multiplier=c.get("candidate_multiplier",20),
                       target_modality=args.target,exclude_id=args.exclude_image_id,
                       exclude_patient=args.exclude_patient_id,query_category=args.label_aware_category,
                       label_aware=args.label_aware_category is not None,
                       alpha=c.get("ranking_alpha",.7), beta=c.get("ranking_beta",.1),
                       gamma=c.get("ranking_gamma",.2))
    write_json(args.out,{"query_modality":args.modality,"target_modality":args.target,
                         "label_aware":args.label_aware_category is not None,"results":results})
    print(json.dumps(results,indent=2))
    print(f"Saved to {Path(args.out).resolve()}")

if __name__=="__main__":
    main()
