#!/usr/bin/env python3
"""Embed an official patient-split repository and build persistent FAISS/exact index."""
import argparse
import time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from qmed.training.trainer import load_checkpoint
from qmed.data.dataset import MedicalDataset
from qmed.retrieval.embedding import extract_embeddings
from qmed.retrieval.faiss_index import RetrievalIndex
from qmed.utils.config import resolve_device,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--manifest")
    p.add_argument("--split",choices=["train","val","test","all"],default="test")
    p.add_argument("--out",default="outputs/index")
    p.add_argument("--index",choices=["faiss_ivf","faiss_flat","exact"])
    p.add_argument("--device",default="auto")
    args=p.parse_args()
    device=resolve_device(args.device)
    model,cfg,_=load_checkpoint(args.checkpoint,device)
    manifest=args.manifest or cfg["data"]["manifest"]
    ds=MedicalDataset(manifest,image_size=cfg["data"]["image_size"],
                      split=args.split if args.split!="all" else None,
                      roi_policy=cfg["data"].get("roi_policy","none"),
                      quality_filter=cfg["data"].get("quality_filter",False),
                      min_quality=cfg["data"].get("min_quality",.35))
    start_embed=time.perf_counter()
    vectors,metadata=extract_embeddings(model,ds,device,cfg["data"]["batch_size"],cfg["data"].get("workers",0))
    embedding_time=time.perf_counter()-start_embed
    if args.index:
        cfg["retrieval"]["index"]=args.index
    start_index=time.perf_counter()
    idx=RetrievalIndex.from_config(cfg).build(vectors,metadata)
    indexing_time=time.perf_counter()-start_index
    destination=idx.save(args.out)
    write_json(destination/"build_profile.json", {"n_images":len(vectors),"embedding_dimension":vectors.shape[1],
               "embedding_seconds":embedding_time,"indexing_seconds":indexing_time,
               "index_actual":idx.engine,"split":args.split,"hardware":str(device)})
    print(f"Saved {len(vectors)} embeddings of dimension {vectors.shape[1]} to {destination.resolve()}; index={idx.engine}")

if __name__=="__main__":
    main()
