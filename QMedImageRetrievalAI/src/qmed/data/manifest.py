"""Dataset manifest ingestion for all three official sources plus leakage-safe splitting."""
from __future__ import annotations
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .uccs import brats_category, xray_category, deep_lesion_category, MODALITIES

REQUIRED = ["image_id", "patient_id", "modality", "uccs", "image_path"]
MRI_SEQUENCES = ["seq_t1", "seq_t1ce", "seq_t2", "seq_flair"]


def _index_files(root: str | Path, extensions=(".png", ".jpg", ".jpeg", ".dcm")) -> dict[str, str]:
    return {p.name: str(p.resolve()) for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in extensions}


def _pick_col(frame, choices):
    mapping = {str(c).lower().replace("_", " ").strip(): c for c in frame.columns}
    for item in choices:
        if item.lower().replace("_", " ") in mapping:
            return mapping[item.lower().replace("_", " ")]
    return None


def _numeric_tokens(x):
    if pd.isna(x) if np.isscalar(x) else False:
        return []
    return [float(v) for v in re.findall(r"[-+]?(?:\d*\.)?\d+(?:[Ee][-+]?\d+)?", str(x))]


def _read_mm(row: pd.Series):
    # Some independently exported DL_info tables provide a measured length in mm.
    for field in ("Measured_long_diameter", "Long_diameter_mm", "diameter_mm", "Long axis", "long_axis_mm"):
        if field in row.index:
            try:
                x = float(row[field])
                if np.isfinite(x) and x > 0:
                    return x
            except (ValueError, TypeError):
                pass
    # Official DeepLesion CSV commonly stores RECIST endpoints and pixel spacing,
    # rather than a precomputed length in millimeters.
    spacing = _numeric_tokens(row.get("Spacing_mm_px_", row.get("Pixel_spacing", np.nan)))
    if spacing:
        # Measurement_coordinates: [x1,y1,x2,y2,x3,y3,x4,y4]. First pair
        # contains long RECIST diameter endpoints. Spacing is [row_mm,col_mm].
        points = _numeric_tokens(row.get("Measurement_coordinates", np.nan))
        if len(points) >= 4 and len(spacing) >= 2:
            dx = (points[2]-points[0])*spacing[1]
            dy = (points[3]-points[1])*spacing[0]
            length = float(np.hypot(dx,dy))
            if 0 < length < 1000:
                return length
        diam = _numeric_tokens(row.get("Lesion_diameters_Pixel_", np.nan))
        if diam:
            # Approximation when only diameter-pixels and anisotropic spacing
            # are available; use geometric mean, and document inferred mm.
            length = float(diam[0]*np.sqrt(spacing[0]*spacing[-1]))
            if 0 < length < 1000:
                return length
    return None  # Never create a UCCS label without defensible mm metadata.


def prepare_deeplesion(root: str | Path, metadata_csv: str | None = None) -> pd.DataFrame:
    root = Path(root)
    metadata_csv = metadata_csv or str(next(iter(root.rglob("DL_info.csv")), ""))
    if not metadata_csv or not Path(metadata_csv).is_file():
        raise FileNotFoundError("DeepLesion requires DL_info.csv (or --deeplesion-csv)")
    df = pd.read_csv(metadata_csv)
    fn = _pick_col(df, ["File_name", "File name", "filename"])
    pat = _pick_col(df, ["Patient_index", "Patient ID", "patient_id"])
    lesion = _pick_col(df, ["Lesion_index", "Lesion ID", "lesion_id"])
    if not fn or not pat:
        raise ValueError("DeepLesion CSV missing File_name or Patient_index column")
    paths = _index_files(root)
    records = []
    for i, row in df.iterrows():
        filename = Path(str(row[fn])).name
        if filename not in paths:
            continue
        diameter = _read_mm(row)
        category = deep_lesion_category(diameter)
        if category is None:
            continue
        records.append({"image_id": f"DL_{i:06d}", "patient_id": f"DL_{row[pat]}",
                        "modality": "CT", "uccs": category, "image_path": paths[filename],
                        "source_dataset": "DeepLesion", "lesion_id": str(row[lesion]) if lesion else str(i),
                        "diameter_mm": diameter, "native_label": "lesion"})
    if not records:
        raise ValueError("No labeled DeepLesion images matched. Provide actual CT files and a diameter_mm column in DL_info.csv if the original export lacks measurements in mm.")
    return pd.DataFrame(records)


def prepare_chestxray(root: str | Path, metadata_csv: str | None = None) -> pd.DataFrame:
    root = Path(root)
    metadata_csv = metadata_csv or str(next(iter(root.rglob("Data_Entry_2017*.csv")), ""))
    if not metadata_csv or not Path(metadata_csv).is_file():
        raise FileNotFoundError("ChestX-ray14 requires Data_Entry_2017*.csv")
    df = pd.read_csv(metadata_csv)
    file_col = _pick_col(df, ["Image Index", "Image_Index", "filename"])
    pat_col = _pick_col(df, ["Patient ID", "Patient_ID"])
    label_col = _pick_col(df, ["Finding Labels", "Finding_Labels"])
    if not all((file_col, pat_col, label_col)):
        raise ValueError("ChestX-ray14 CSV missing Image Index / Patient ID / Finding Labels")
    paths = _index_files(root)
    records = []
    for _, r in df.iterrows():
        name = str(r[file_col])
        category = xray_category(str(r[label_col]))
        if name not in paths or category is None:
            continue
        records.append({"image_id": f"XR_{name}", "patient_id": f"XR_{r[pat_col]}",
                        "modality": "XRAY", "uccs": category,
                        "image_path": paths[name], "native_label": str(r[label_col]),
                        "source_dataset": "ChestXray14"})
    if not records:
        raise ValueError("No eligible X-ray images matched the metadata CSV")
    return pd.DataFrame(records)


def prepare_brats(root: str | Path, max_slices_per_patient: int = 50, seed: int = 0) -> pd.DataFrame:
    try:
        import nibabel as nib
    except ImportError as error:
        raise ImportError("BraTS requires nibabel: pip install -e '.[medical]'") from error
    root = Path(root)
    rng = np.random.default_rng(seed)
    records = []
    for seg_path in sorted(root.rglob("*_seg.nii*")):
        subject = seg_path.name.split("_seg.nii")[0]
        folder = seg_path.parent
        seq_files = {}
        for seq in ("t1", "t1ce", "t2", "flair"):
            choices = list(folder.glob(subject + "_" + seq + ".nii*"))
            if not choices:
                break
            seq_files["seq_" + seq] = str(choices[0].resolve())
        if len(seq_files) != 4:
            warnings.warn(f"Skipping BraTS subject {subject}: incomplete sequences")
            continue
        mask_vol = np.asanyarray(nib.load(str(seg_path)).dataobj)
        if mask_vol.ndim != 3:
            raise ValueError(f"BraTS seg is not a volume: {seg_path}")
        # Eligible anatomical slices are those with nonzero brain signal in FLAIR;
        # tumour-free axial slices remain eligible for C5.
        flair = nib.load(seq_files["seq_flair"])
        eligible = []
        for z in range(mask_vol.shape[2]):
            foreground = np.asarray(flair.dataobj[..., z])
            if np.count_nonzero(foreground) < foreground.size * .04:
                continue
            eligible.append((z, brats_category(mask_vol[..., z])))
        if max_slices_per_patient and len(eligible) > max_slices_per_patient:
            # balanced deterministic sampling, then complete with random eligible slices
            chosen = []
            per_class = max(1, max_slices_per_patient // 3)
            for cls in (0, 2, 4):
                positions = [x for x in eligible if x[1] == cls]
                if positions:
                    take = min(len(positions), per_class)
                    selected = rng.choice(len(positions), take, replace=False)
                    chosen.extend(positions[int(j)] for j in selected)
            remaining = [x for x in eligible if x not in chosen]
            need = min(max_slices_per_patient - len(chosen), len(remaining))
            if need > 0:
                chosen.extend(remaining[int(j)] for j in rng.choice(len(remaining), need, replace=False))
            eligible = sorted(chosen)
        for z, cls in eligible:
            records.append({"image_id": f"BT_{subject}_{z:03d}", "patient_id": f"BT_{subject}",
                            "modality": "MRI", "uccs": cls, "slice_idx": z,
                            "image_path": seq_files["seq_flair"], "source_dataset": "BraTS2020",
                            "native_label": "mask_based_category", "seg_path": str(seg_path.resolve()),
                            **seq_files})
    if not records:
        raise ValueError("No eligible BraTS subjects; expected nested '*_seg.nii.gz' and four MRI volumes")
    return pd.DataFrame(records)


def validate_manifest(df: pd.DataFrame, validate_paths: bool = False) -> pd.DataFrame:
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    df = df.copy()
    df["modality"] = df["modality"].str.upper().replace({"X-RAY": "XRAY", "X_RAY": "XRAY"})
    if not df["modality"].isin(MODALITIES).all():
        raise ValueError("modality must be CT, MRI, XRAY")
    df["uccs"] = pd.to_numeric(df["uccs"], errors="raise").astype(int)
    if not df["uccs"].isin(range(5)).all():
        raise ValueError("UCCS categories must be integers 0..4 (C1..C5)")
    if df["image_id"].duplicated().any():
        raise ValueError("image_id duplicates present; IDs must be unique even with multiple ROIs")
    if validate_paths:
        for col in ["image_path"] + [x for x in MRI_SEQUENCES if x in df]:
            missing_paths = [str(x) for x in df[col].dropna() if str(x).strip() and not Path(str(x)).is_file()]
            if missing_paths:
                raise FileNotFoundError(f"{len(missing_paths)} missing {col} paths, e.g. {missing_paths[0]}")
    return df


def patient_level_split(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """70:15:15 group split, separately by modality. Samples from one patient never cross splits."""
    df = validate_manifest(df)
    result = df.copy()
    result["split"] = ""
    for modality, group in result.groupby("modality"):
        patients = group["patient_id"].astype(str)
        if patients.nunique() < 5:
            raise ValueError(f"{modality}: need >=5 distinct patients for train/val/test group splits")
        gss1 = GroupShuffleSplit(n_splits=1, test_size=.30, random_state=seed)
        train_idx, held_idx = next(gss1.split(group, groups=patients))
        held = group.iloc[held_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=.50, random_state=seed + 1)
        val_idx, test_idx = next(gss2.split(held, groups=held["patient_id"].astype(str)))
        result.loc[group.iloc[train_idx].index, "split"] = "train"
        result.loc[held.iloc[val_idx].index, "split"] = "val"
        result.loc[held.iloc[test_idx].index, "split"] = "test"
    for group in result.groupby("patient_id"):
        if group[1]["split"].nunique() > 1:
            raise AssertionError(f"patient leakage: {group[0]}")
    return result
