#!/usr/bin/env python3
"""Create manuscript-ready 600-dpi plots from ACTUAL experiment CSVs."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from qmed.retrieval.faiss_index import RetrievalIndex


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-dir",help="saved embeddings and metadata for PCA figure")
    p.add_argument("--ablation-summary",help="outputs/ablations/ablation_summary.csv")
    p.add_argument("--retrieval-summary",help="outputs/evaluation/retrieval_summary.csv")
    p.add_argument("--out",default="outputs/figures")
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.index_dir:
        idx=RetrievalIndex.load(args.index_dir)
        xy=PCA(n_components=2).fit_transform(idx.vectors)
        fig,ax=plt.subplots(figsize=(6.8,5))
        for mod in sorted(idx.metadata.modality.unique()):
            mask=idx.metadata.modality.to_numpy()==mod
            ax.scatter(xy[mask,0],xy[mask,1],s=15,alpha=.65,label=mod)
        ax.set_xlabel("Principal component 1")
        ax.set_ylabel("Principal component 2")
        ax.legend(loc="best",frameon=False)
        fig.tight_layout();fig.savefig(out/"embedding_pca.png",dpi=600,bbox_inches="tight");plt.close(fig)
    if args.ablation_summary:
        table=pd.read_csv(args.ablation_summary).dropna(subset=['mAP_mean'])
        fig,ax=plt.subplots(figsize=(max(6, .75*len(table)),4.7))
        ax.bar(np.arange(len(table)),table.mAP_mean,yerr=table.mAP_sd.fillna(0).to_numpy(),capsize=3)
        ax.set_xticks(np.arange(len(table)),table.variant,rotation=35,ha="right")
        ax.set_ylabel("Mean average precision (mAP)")
        ax.set_ylim(0,1);fig.tight_layout();fig.savefig(out/"ablation_map.png",dpi=600,bbox_inches="tight");plt.close(fig)
    if args.retrieval_summary:
        table=pd.read_csv(args.retrieval_summary).dropna(subset=['mAP'])
        labels=table.source.astype(str)+"→"+table.target.astype(str)
        fig,ax=plt.subplots(figsize=(7,4.2))
        ax.bar(labels,table.mAP)
        ax.set_ylabel("Mean average precision (mAP)")
        ax.set_xlabel("Retrieval direction")
        ax.set_ylim(0,1)
        ax.tick_params(axis="x",rotation=45)
        fig.tight_layout();fig.savefig(out/"retrieval_directions.png",dpi=600,bbox_inches="tight");plt.close(fig)
    print(f"Saved requested figures to {out.resolve()}")

if __name__=="__main__":
    main()
