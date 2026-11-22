"""Per-seed paired tests, patient-cluster bootstrap, Wilcoxon/Holm correction."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def paired_seed_statistics(proposed: list[float], baseline: list[float]):
    proposed = np.asarray(proposed, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if len(proposed) != len(baseline) or len(proposed) < 2:
        raise ValueError("paired statistics require >=2 matched seeds")
    difference = proposed - baseline
    mean_diff = float(np.mean(difference))
    sem = float(stats.sem(difference))
    df = len(difference)-1
    ci = stats.t.interval(.95, df, loc=mean_diff, scale=sem)
    t = stats.ttest_rel(proposed, baseline)
    try:
        wilcoxon = stats.wilcoxon(difference, zero_method="wilcox")
        wp = float(wilcoxon.pvalue)
    except ValueError:
        wp = 1.0
    effect_size = mean_diff / float(np.std(difference, ddof=1)) if np.std(difference, ddof=1) > 0 else float("nan")
    return {"n_seeds":len(difference), "mean_proposed":float(np.mean(proposed)),
            "sd_proposed":float(np.std(proposed,ddof=1)), "mean_baseline":float(np.mean(baseline)),
            "sd_baseline":float(np.std(baseline,ddof=1)), "mean_paired_difference":mean_diff,
            "ci95_low":float(ci[0]), "ci95_high":float(ci[1]),
            "t_pvalue":float(t.pvalue), "wilcoxon_pvalue":wp,
            "cohens_dz":float(effect_size)}


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    result = np.empty_like(p)
    current = 0.0
    for rank, idx in enumerate(order):
        current = max(current, min(1.0, float(p[idx])*(len(p)-rank)))
        result[idx] = current
    return result.tolist()


def cluster_bootstrap_mean(per_query: pd.DataFrame, metric="AP", clusters="patient_id", repeats=10000, seed=0):
    """Patient-cluster bootstrap avoids falsely independent 2D slices."""
    if per_query.empty:
        raise ValueError("no per-query evaluation rows")
    groups = [chunk[metric].to_numpy(dtype=float) for _,chunk in per_query.groupby(clusters, sort=True)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats)
    n = len(groups)
    for i in range(repeats):
        sampled = rng.integers(0, n, size=n)
        parts = [groups[j] for j in sampled]
        estimates[i] = np.concatenate(parts).mean()
    ci = np.quantile(estimates, [.025,.975])
    return {"mean":float(per_query[metric].mean()), "low":float(ci[0]),
            "high":float(ci[1]), "clusters":n, "bootstrap_repeats":repeats}


def per_query_paired_wilcoxon(a: pd.DataFrame, b: pd.DataFrame, metric="AP"):
    matched = a[["query_id",metric]].merge(b[["query_id",metric]], on="query_id", suffixes=("_proposed", "_baseline"))
    if len(matched) < 2:
        raise ValueError("Not enough matched queries for paired Wilcoxon")
    diffs = matched[f"{metric}_proposed"] - matched[f"{metric}_baseline"]
    if np.allclose(diffs.to_numpy(),0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(diffs).pvalue)
    return {"n_queries":len(matched), "mean_delta":float(diffs.mean()), "pvalue":p}
