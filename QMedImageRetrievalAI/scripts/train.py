#!/usr/bin/env python3
"""End-to-end training for patient-disjoint UCCS-conditioned retrieval."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from qmed.training.trainer import train_model
from qmed.utils.config import load_config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/default.yaml")
    p.add_argument("--manifest",help="Override manifest CSV")
    p.add_argument("--out",default="outputs/train")
    p.add_argument("--seed",type=int)
    p.add_argument("--resume",help="Resume from a trusted last.pt")
    p.add_argument("--override",action="append",default=[],help="dot.path=yamlvalue, e.g. model.quantum_backend=none")
    args=p.parse_args()
    import yaml
    overrides={}
    for x in args.override:
        key,value=x.split("=",1)
        overrides[key]=yaml.safe_load(value)
    cfg=load_config(args.config,overrides)
    if args.seed is not None:
        cfg["seed"]=args.seed
    if args.manifest:
        cfg["data"]["manifest"]=args.manifest
    path=train_model(cfg,args.out,resume=args.resume)
    print(f"BEST CHECKPOINT: {path.resolve()}")

if __name__ == "__main__":
    main()
