"""Batch embedding extraction for a training/validation/test set."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from qmed.data.dataset import MedicalDataset


@torch.inference_mode()
def extract_embeddings(model, ds: MedicalDataset, device, batch_size=32, workers=0):
    model.eval()
    iterator = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    vectors, rows = [], []
    for batch in iterator:
        emb = model(batch["image"].to(device), batch["modality"].to(device))
        vectors.append(emb.cpu().numpy().astype(np.float32))
        for j in range(len(batch["image_id"])):
            rows.append({
                "image_id": batch["image_id"][j], "patient_id": batch["patient_id"][j],
                "modality": ("CT", "MRI", "XRAY")[int(batch["modality"][j])],
                "uccs": int(batch["uccs"][j]), "quality": float(batch["quality"][j]),
                "image_path": str(ds.frame.iloc[len(rows)]["image_path"]),
            })
    return np.concatenate(vectors).astype(np.float32), pd.DataFrame(rows)
