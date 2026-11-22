#!/usr/bin/env python3
"""Save non-diagnostic Grad-CAM heatmap for similarity to a retrieved image."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qmed.training.trainer import load_checkpoint
from qmed.data.preprocessing import normalize_image,read_grayscale
from qmed.data.dataset import MODALITY_TO_ID
from qmed.evaluation.explainability import similarity_gradcam

def image_input(path,side,device):
    mono=normalize_image(read_grayscale(path),side)
    return torch.cat([mono]+[torch.zeros_like(mono)]*3,0)[None].to(device)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--query',required=True)
    p.add_argument('--retrieved',required=True)
    p.add_argument('--query-modality',choices=['CT','MRI','XRAY'],required=True)
    p.add_argument('--retrieved-modality',choices=['CT','MRI','XRAY'],required=True)
    p.add_argument('--out',default='outputs/gradcam.png')
    args=p.parse_args()
    model,cfg,_=load_checkpoint(args.checkpoint,'cpu')
    side=cfg['data']['image_size']
    q=image_input(args.query,side,'cpu')
    r=image_input(args.retrieved,side,'cpu')
    qmod=torch.tensor([MODALITY_TO_ID[args.query_modality]])
    rmod=torch.tensor([MODALITY_TO_ID[args.retrieved_modality]])
    heat,score=similarity_gradcam(model,q,qmod,r,rmod)
    background=q[0,0].numpy()
    fig,ax=plt.subplots(figsize=(4.5,4.5))
    ax.imshow(background,cmap='gray')
    ax.imshow(heat,cmap='jet',alpha=.36)
    ax.axis('off')
    fig.tight_layout(pad=0)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=600,bbox_inches='tight',pad_inches=0)
    plt.close(fig)
    print(f'Saved retrieval-similarity heatmap {path.resolve()}, cosine score={score:.4f}. This is NOT diagnostic validation.')

if __name__=='__main__':
    main()
