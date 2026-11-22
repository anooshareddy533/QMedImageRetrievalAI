"""Lazy multi-modal grayscale loading from a validated unified manifest."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from .manifest import MRI_SEQUENCES, validate_manifest
from .preprocessing import read_grayscale, normalize_image, augment, apply_roi, quality_score

MODALITY_TO_ID = {"CT": 0, "MRI": 1, "XRAY": 2}
ID_TO_MODALITY = {v:k for k,v in MODALITY_TO_ID.items()}


class MedicalDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame | str, image_size: int = 224,
                 split: str | None = None, training: bool = False,
                 roi_policy: str = "none", quality_filter: bool = False,
                 min_quality: float = 0.35):
        source_path = Path(manifest) if isinstance(manifest, (str, Path)) else None
        if source_path is not None:
            df = pd.read_csv(source_path)
            relative_path_columns = ["image_path", "roi_mask_path", *MRI_SEQUENCES]
            for col in relative_path_columns:
                if col in df:
                    df[col] = df[col].apply(lambda p: str((source_path.parent / str(p)).resolve())
                                            if pd.notna(p) and str(p).strip() and not Path(str(p)).is_absolute() else p)
        else:
            df = manifest.copy()
        df = validate_manifest(df)
        if quality_filter:
            if "quality" not in df.columns:
                raise ValueError("quality_filter requires a precomputed 'quality' column. Run scripts/quality_screen.py first.")
            df = df.loc[pd.to_numeric(df["quality"], errors="coerce") >= float(min_quality)]
        if split is not None:
            if "split" not in df:
                raise ValueError("manifest has no split column; run scripts/prepare_data.py")
            df = df.loc[df["split"] == split]
        self.frame = df.reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"Empty medical dataset: split={split}")
        self.image_size = int(image_size)
        self.training = training
        self.roi_policy = roi_policy
        self.quality_filter = quality_filter
        self.min_quality = float(min_quality)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        z = int(row.get("slice_idx", 0)) if pd.notna(row.get("slice_idx", 0)) else 0
        if row.modality == "MRI" and all(col in self.frame and pd.notna(row.get(col)) for col in MRI_SEQUENCES):
            channels = [normalize_image(read_grayscale(str(row[col]), z), self.image_size)
                        for col in MRI_SEQUENCES]
            image = torch.cat(channels, dim=0)
        else:
            mono = normalize_image(read_grayscale(str(row.image_path), z), self.image_size)
            image = torch.cat([mono, torch.zeros_like(mono), torch.zeros_like(mono), torch.zeros_like(mono)], dim=0)
        if self.roi_policy == "supplied" and pd.notna(row.get("roi_mask_path")):
            mask = normalize_image(read_grayscale(str(row.roi_mask_path), z), self.image_size)
            image = apply_roi(image, mask)
        elif self.roi_policy not in ("none", "supplied"):
            raise ValueError(f"ROI policy must be 'none' or 'supplied', got {self.roi_policy}")
        quality = quality_score(image[:1])
        # Filtering dynamically would alter dataset lengths and bias metrics; screen offline.
        if self.training:
            image = augment(image)
        return {
            "image": image,
            "modality": torch.tensor(MODALITY_TO_ID[str(row.modality)], dtype=torch.long),
            "uccs": torch.tensor(int(row.uccs), dtype=torch.long),
            "image_id": str(row.image_id),
            "patient_id": str(row.patient_id),
            "quality": torch.tensor(quality, dtype=torch.float32),
        }


def balanced_sampler(ds: MedicalDataset):
    # Inverse frequency of (modality, UCCS) groups; avoid dominating CT/X-ray.
    keys = ds.frame["modality"].astype(str) + ":" + ds.frame["uccs"].astype(str)
    freqs = keys.value_counts()
    weights = np.array([1.0 / freqs[x] for x in keys], dtype=np.float64)
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(ds), replacement=True)


def loader_from_config(manifest, config, split, shuffle=False):
    from torch.utils.data import DataLoader
    ds = MedicalDataset(manifest, image_size=config["data"]["image_size"], split=split,
                        training=(split == "train"), roi_policy=config["data"].get("roi_policy", "none"),
                        quality_filter=config["data"].get("quality_filter", False),
                        min_quality=config["data"].get("min_quality", .35))
    sampler = balanced_sampler(ds) if shuffle and config["training"].get("balanced_sampling", True) else None
    return DataLoader(ds, batch_size=config["data"]["batch_size"], shuffle=shuffle and sampler is None,
                      sampler=sampler, num_workers=config["data"].get("workers", 0), pin_memory=torch.cuda.is_available())
