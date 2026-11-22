#!/usr/bin/env python3
"""Compute documented sharpness/contrast/signal proxy and optionally filter manifest."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
from qmed.data.preprocessing import read_grayscale,normalize_image,quality_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--size',type=int,default=224)
    p.add_argument('--threshold',type=float,help='Optional explicitly validated threshold, e.g. 0.35')
    args=p.parse_args()
    df=pd.read_csv(args.manifest)
    scores=[]
    for _,row in df.iterrows():
        z=int(row.get('slice_idx',0)) if pd.notna(row.get('slice_idx',0)) else 0
        filepath=row['image_path']
        if not Path(str(filepath)).is_absolute():
            filepath=str((Path(args.manifest).parent/str(filepath)).resolve())
        image=normalize_image(read_grayscale(filepath,z),args.size)
        scores.append(quality_score(image))
    df['quality']=scores
    n_before=len(df)
    if args.threshold is not None:
        df=df.loc[df['quality']>=args.threshold]
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(dest,index=False)
    print(f'Quality scoring complete: {n_before} inputs, {len(df)} retained, score range=[{min(scores):.3f},{max(scores):.3f}] -> {dest}')
    print('Thresholds should be chosen on validation data, then frozen before test evaluation.')

if __name__=='__main__':
    main()
