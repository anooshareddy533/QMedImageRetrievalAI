#!/usr/bin/env python3
"""Create synthetic toy fixtures for SOFTWARE TESTS ONLY (never clinical evidence)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from qmed.data.manifest import patient_level_split


def paint(category, modality, size, seed):
    rng = np.random.default_rng(seed)
    image = Image.new("L", (size,size), color=25 if modality == "CT" else 35)
    pen = ImageDraw.Draw(image)
    pad = size//10
    pen.ellipse([pad,pad,size-pad,size-pad], outline=110, width=max(1,size//20), fill=50)
    if category == 0:
        pen.ellipse([size*.28,size*.27,size*.74,size*.70], fill=210)
    elif category == 1:
        pen.ellipse([size*.43,size*.41,size*.60,size*.56], fill=215)
    elif category == 2:
        pen.ellipse([size*.2,size*.35,size*.85,size*.77], fill=125)
    elif category == 3:
        pen.line([size*.2,size*.5,size*.8,size*.5], fill=220, width=max(2,size//9))
    else:
        pen.ellipse([size*.34,size*.34,size*.66,size*.66], outline=65, width=2)
    base = np.asarray(image,dtype=np.float32)
    noise = rng.normal(0,7,size=(size,size)).astype(np.float32)
    return np.uint8(np.clip(base+noise,0,255))


def generate(output, patients_per_category=6, size=64, seed=0):
    folder = Path(output)
    images = folder / "images"
    images.mkdir(parents=True, exist_ok=True)
    records=[]
    groups={"CT":[0,1],"MRI":[0,2,4],"XRAY":[0,1,2,3,4]}
    counter=0
    for modality, categories in groups.items():
        for category in categories:
            for patient in range(patients_per_category):
                pid = f"FAKE_{modality}_{category}_{patient:02d}"
                image_id = f"FAKE_{counter:05d}"
                array = paint(category, modality, size, seed+counter*17)
                imgpath = images / f"{image_id}.png"
                Image.fromarray(array).save(imgpath)
                entry={"image_id":image_id,"patient_id":pid,"modality":modality,
                       "uccs":category,"image_path":str(imgpath.resolve()),
                       "source_dataset":"SYNTHETIC_SOFTWARE_TEST_ONLY", "native_label":f"toy_{category}"}
                if modality == "MRI":
                    for j, seq in enumerate(("seq_t1","seq_t1ce","seq_t2","seq_flair")):
                        v = np.uint8(np.clip(array.astype(np.float32)*(.9+.05*j)+8*j,0,255))
                        seq_path = images / f"{image_id}_{seq}.png"
                        Image.fromarray(v).save(seq_path)
                        entry[seq] = str(seq_path.resolve())
                records.append(entry)
                counter+=1
    df = patient_level_split(pd.DataFrame(records), seed=seed)
    target = folder/"manifest.csv"
    df.to_csv(target,index=False)
    print(f"SYNTHETIC TEST DATA ONLY: saved {len(df)} fake records to {target.resolve()}")
    print(pd.crosstab(df.modality,df.split).to_string())
    return target

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="outputs/smoke")
    parser.add_argument("--patients-per-category",type=int,default=6)
    parser.add_argument("--size",type=int,default=64)
    parser.add_argument("--seed",type=int,default=0)
    args=parser.parse_args()
    generate(args.output,args.patients_per_category,args.size,args.seed)
