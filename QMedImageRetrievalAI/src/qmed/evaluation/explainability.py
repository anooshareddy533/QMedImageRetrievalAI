"""Retrieval similarity Grad-CAM: research visualization, NOT radiologist evidence."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def similarity_gradcam(model, query_batch, query_mod, target_batch, target_mod):
    """Backprop cosine(query,target) into query ResNet/Tiny layer4 activations.

    Returns averaged class activation mask [H,W]. Multi-sequence MRI activations
    are averaged; does not claim to localize verified pathology.
    """
    model.eval()
    activations=[]
    gradients=[]
    def store_activation(_module,_input,output):
        if output.requires_grad:
            activations.append(output)
            output.register_hook(lambda grad: gradients.append(grad))
    hook=model.backbone.layer4.register_forward_hook(store_activation)
    try:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            target_emb=model(target_batch,target_mod)
        query_emb=model(query_batch,query_mod)
        score=(query_emb*target_emb).sum()
        score.backward()
        if not activations or not gradients:
            raise RuntimeError("CNN layer4 activation gradients unavailable")
        a=activations[-1].detach()
        g=gradients[-1].detach()
        weights=g.mean(dim=(-2,-1),keepdim=True)
        heat=F.relu((a*weights).sum(dim=1,keepdim=True))
        heat=F.interpolate(heat,size=query_batch.shape[-2:],mode='bilinear',align_corners=False)
        heat=heat.mean(dim=0)[0]
        heat=heat/(heat.max()+1e-10)
        return heat.cpu().numpy(),float(score.detach().cpu())
    finally:
        hook.remove()
