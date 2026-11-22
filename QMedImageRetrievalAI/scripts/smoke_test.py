#!/usr/bin/env python3
"""One-command end-to-end integration smoke test (synthetic images only)."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def run(*args):
    env=os.environ.copy()
    env.setdefault("OMP_NUM_THREADS","2")
    env.setdefault("MKL_NUM_THREADS","2")
    cmd=[sys.executable,str(ROOT/"scripts"/args[0]),*map(str,args[1:])]
    print("+", " ".join(cmd),flush=True)
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)


def main():
    folder=ROOT/"outputs"/"smoke"
    run("make_synthetic_fixtures.py","--output",folder,"--patients-per-category",6,"--size",64)
    run("train.py","--config",ROOT/"configs"/"smoke.yaml","--out",folder/"train")
    checkpoint=folder/"train"/"best.pt"
    assert checkpoint.is_file()
    run("build_index.py","--checkpoint",checkpoint,"--manifest",folder/"manifest.csv",
        "--split","test","--out",folder/"index","--device","cpu")
    run("evaluate.py","--index-dir",folder/"index","--out",folder/"evaluation","--k",5,10)
    with (folder/"evaluation"/"overall.json").open() as f:
        metrics=json.load(f)
    assert metrics["n_queries"]>0 and 0 <= metrics["mAP"] <= 1
    import pandas as pd
    test_set=pd.read_csv(folder/"manifest.csv")
    row=test_set[(test_set.split=="test") & (test_set.modality=="CT")].iloc[0]
    run("retrieve.py","--checkpoint",checkpoint,"--index-dir",folder/"index",
        "--image",row.image_path,"--modality","CT","--k",3,
        "--exclude-image-id",row.image_id,"--out",folder/"query.json","--device","cpu")
    with (folder/"query.json").open() as f:
        results=json.load(f)["results"]
    assert results and all(x["image_id"]!=row.image_id for x in results)
    print("PASS: synthetic multi-modal training, differentiable 8-qubit VQC, checkpoint, index, exact metrics and query search",flush=True)
    print("WARNING: THESE RESULTS ARE SYNTHETIC SOFTWARE TESTS, NOT MEDICAL IMAGE EXPERIMENTS")

if __name__=="__main__":
    main()
