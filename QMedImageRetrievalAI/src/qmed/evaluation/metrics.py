"""Exact retrieval evaluation with full-gallery AP; no mislabeled ANN/truncated mAP."""
from __future__ import annotations
import numpy as np
import pandas as pd


def evaluate_embeddings(embeddings: np.ndarray, metadata: pd.DataFrame,
                        k_values=(5,10,20), source=None, target=None,
                        max_queries=0, exclude_same_patient=True,
                        seed=0, save_per_query: str | None = None):
    """Every eligible query is compared with the entire eligible gallery by cosine.

    No target labels affect ranking. Category labels appear *only* in the relevance
    judgment. Full AP denominator is the number of relevant gallery entries.
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10)
    n = len(emb)
    if len(metadata) != n:
        raise ValueError("embeddings and metadata row counts differ")
    mods = metadata["modality"].to_numpy()
    labels = metadata["uccs"].to_numpy().astype(int)
    pids = metadata["patient_id"].astype(str).to_numpy()
    ids = metadata["image_id"].astype(str).to_numpy()
    source_indices = np.where(mods == source)[0] if source else np.arange(n)
    gallery_indices = np.where(mods == target)[0] if target else np.arange(n)
    if max_queries and len(source_indices) > max_queries:
        source_indices = np.sort(np.random.default_rng(seed).choice(source_indices, size=max_queries, replace=False))
    maxk = max(map(int, k_values))
    rows = []
    gal_emb = emb[gallery_indices]
    gal_labels = labels[gallery_indices]
    gal_ids = ids[gallery_indices]
    gal_pids = pids[gallery_indices]
    for qi in source_indices:
        eligible = gal_ids != ids[qi]
        if exclude_same_patient:
            eligible &= gal_pids != pids[qi]
        # Exclusion is applied to BOTH the rankings and the AP denominator.
        hits = gal_labels == labels[qi]
        relevant_total = int(np.count_nonzero(hits & eligible))
        if relevant_total == 0:
            continue
        scores = gal_emb @ emb[qi]
        scores[~eligible] = -np.inf
        order = np.argsort(-scores, kind="stable")
        order = order[eligible[order]]
        relevances = hits[order].astype(np.int64)
        n_rel = len(relevances)
        cumulative = np.cumsum(relevances)
        ranks = np.arange(1, n_rel+1)
        ap = float(((cumulative / ranks) * relevances).sum() / relevant_total)
        relevant_positions = np.flatnonzero(relevances)
        reciprocal_rank = 1.0/(int(relevant_positions[0])+1)
        row = {"query_id": ids[qi], "patient_id": pids[qi], "source": mods[qi],
               "target": target or "ALL", "uccs": labels[qi], "AP": ap,
               "MRR": reciprocal_rank, "eligible_gallery": len(order), "relevant_gallery": relevant_total}
        for k in k_values:
            k = int(k)
            n_hits = int(relevances[:k].sum())
            p = n_hits / k  # standard P@K denominator even if gallery<K
            r = n_hits / relevant_total
            f1 = 2*p*r/(p+r) if p+r > 0 else 0.0
            dcg = float((relevances[:k] / np.log2(np.arange(2, min(k,n_rel)+2))).sum())
            ideal = float((np.ones(min(k,relevant_total)) / np.log2(np.arange(2, min(k,relevant_total)+2))).sum())
            row.update({f"P@{k}": p, f"R@{k}": r, f"F1@{k}": f1,
                        f"NDCG@{k}": dcg/ideal if ideal else 0.0})
        rows.append(row)
    per = pd.DataFrame(rows)
    if save_per_query:
        from pathlib import Path
        Path(save_per_query).parent.mkdir(parents=True, exist_ok=True)
        per.to_csv(save_per_query, index=False)
    if per.empty:
        return {"n_queries": 0, "n_gallery": len(gallery_indices), "mAP": float("nan"),
                "source": source or "ALL", "target": target or "ALL",
                "exclusion": "same patient" if exclude_same_patient else "same image",
                "warning": "No queries with relevant eligible gallery items"}, per
    keys = ["AP", "MRR"] + [f"{s}@{k}" for k in k_values for s in ("P", "R", "F1", "NDCG")]
    summary = {"n_queries": len(per), "n_gallery": len(gallery_indices), "mAP": float(per.AP.mean()),
               "MRR": float(per.MRR.mean())}
    summary.update({k: float(per[k].mean()) for k in keys if k not in ("AP", "MRR")})
    summary["source"] = source or "ALL"
    summary["target"] = target or "ALL"
    summary["exclusion"] = "same patient" if exclude_same_patient else "same image"
    return summary, per


def evaluate_all_directions(embeddings, metadata, k_values=(5,10,20), **kwargs):
    directions = [(m,m) for m in ("CT", "MRI", "XRAY")]
    directions += [(a,b) for a in ("CT", "MRI", "XRAY") for b in ("CT", "MRI", "XRAY") if a != b]
    summaries, frames = [], []
    for source, target in directions:
        if source not in metadata.modality.values or target not in metadata.modality.values:
            continue
        summary, per = evaluate_embeddings(embeddings, metadata, k_values=k_values, source=source, target=target, **kwargs)
        summaries.append(summary)
        if not per.empty:
            frames.append(per)
    return pd.DataFrame(summaries), pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
