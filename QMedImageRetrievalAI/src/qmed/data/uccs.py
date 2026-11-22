"""Paper Section 4.1.1 / Table 3A: reproducible concept-level relevance."""
from __future__ import annotations
import math

CATEGORIES = {
    0: "C1 Mass-like lesion", 1: "C2 Nodular/focal lesion",
    2: "C3 Fluid/oedema-related", 3: "C4 Other structural abnormality",
    4: "C5 No finding",
}
CATEGORY_CODES = ["C1", "C2", "C3", "C4", "C5"]
MODALITIES = ("CT", "MRI", "XRAY")
VALID_BY_MODALITY = {"CT": {0, 1}, "MRI": {0, 2, 4}, "XRAY": {0, 1, 2, 3, 4}}


def deep_lesion_category(diameter_mm: float | str | None) -> int | None:
    if diameter_mm is None or str(diameter_mm).strip() in ("", "nan", "None"):
        return None
    d = float(diameter_mm)
    if not math.isfinite(d) or d <= 0:
        return None
    return 0 if d >= 30.0 else 1


def brats_category(segmentation_slice) -> int:
    """BraTS 2020: label 4 (enhancing) or 1 (core) => C1; label 2 only => C3."""
    import numpy as np
    labels = set(np.unique(segmentation_slice).astype(int).tolist())
    if labels.intersection({1, 4}):
        return 0
    if 2 in labels:
        return 2
    return 4


def xray_category(finding_labels: str) -> int | None:
    labels = {s.strip().lower() for s in str(finding_labels).split("|") if s.strip()}
    if "mass" in labels:
        return 0
    if "nodule" in labels:
        return 1
    if labels.intersection({"effusion", "edema", "oedema"}):
        return 2
    if "no finding" in labels and len(labels) == 1:
        return 4
    if labels - {"no finding"}:
        return 3
    return None


def shared_categories(src: str, dst: str) -> set[int]:
    return VALID_BY_MODALITY[src.upper()] & VALID_BY_MODALITY[dst.upper()]
