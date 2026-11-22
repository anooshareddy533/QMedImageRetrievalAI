"""Fully differentiable category and cross-modal retrieval losses."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _safe_mean(x, reference):
    return x.mean() if x.numel() else reference.sum()*0.0


def retrieval_losses(embeddings, categories, modalities, config):
    """Batch-hard triplets, balanced margin contrastive, UCCS alignment, centroid reg."""
    t = config["training"]
    emb = F.normalize(embeddings, dim=-1)
    dist = torch.cdist(emb, emb, p=2).clamp_min(0.0)
    size = emb.shape[0]
    eye = torch.eye(size, dtype=torch.bool, device=emb.device)
    same = categories[:,None] == categories[None,:]
    same_mod = modalities[:,None] == modalities[None,:]
    pos = same & ~eye
    neg = ~same
    cross_pos = pos & ~same_mod
    upper = torch.triu(torch.ones_like(eye), diagonal=1)
    pos_vals = dist[pos & upper]
    neg_vals = dist[neg & upper]
    pos_loss = _safe_mean(pos_vals.square(), emb)
    neg_loss = _safe_mean(torch.relu(float(t.get("contrastive_margin", 1.0))-neg_vals).square(), emb)
    con = .5 * (pos_loss + neg_loss)

    # Hardest positive and hardest negative for anchors having both.
    valid = pos.any(dim=1) & neg.any(dim=1)
    if valid.any():
        hardest_pos = dist.masked_fill(~pos, float("-inf")).max(dim=1).values[valid]
        hardest_neg = dist.masked_fill(~neg, float("inf")).min(dim=1).values[valid]
        tri = torch.relu(hardest_pos - hardest_neg + float(t.get("triplet_margin", .3))).mean()
    else:
        tri = emb.sum()*0.0
    alignment = _safe_mean(dist[cross_pos & upper].square(), emb)

    # Batch-level centroid compactness. Prevents learning centroids from validation/test.
    reg_terms = []
    for label in torch.unique(categories):
        group = emb[categories == label]
        if len(group) >= 2:
            center = group.mean(dim=0)
            reg_terms.append((group - center).square().sum(dim=1).mean())
    reg = torch.stack(reg_terms).mean() if reg_terms else emb.sum()*0.0
    weights = {
        "alignment": float(t.get("lambda_alignment", .5)),
        "contrastive": float(t.get("lambda_contrastive", 1.0)),
        "triplet": float(t.get("lambda_triplet", 1.0)),
        "regularization": float(t.get("lambda_regularization", .1)),
    }
    parts = {"alignment": alignment, "contrastive": con, "triplet": tri, "regularization": reg}
    total = sum(weights[key] * part for key,part in parts.items())
    return total, {key: float(value.detach().item()) for key,value in parts.items()}
