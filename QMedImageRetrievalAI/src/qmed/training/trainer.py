"""Training, checkpointing, validation mAP early stopping, and reproducibility."""
from __future__ import annotations
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from qmed.data.dataset import loader_from_config, MedicalDataset
from qmed.evaluation.metrics import evaluate_embeddings
from qmed.models.retrieval_model import QMedImageRetrievalAI
from qmed.models.baselines import ClassicalRetrievalBaseline
from qmed.retrieval.embedding import extract_embeddings
from qmed.training.losses import retrieval_losses
from qmed.utils.config import seed_everything, resolve_device, write_json


def train_model(cfg: dict, out_dir: str | Path, manifest: str | Path | pd.DataFrame | None = None,
                resume: str | Path | None = None, verbose=True):
    """Train end-to-end; returns path to best validation-mAP checkpoint.

    The smoke configuration can limit mini-batches for a quick functional check.
    Full research runs use all batches and selected validation queries every epoch.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", cfg)
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    manifest = manifest if manifest is not None else cfg["data"]["manifest"]
    if isinstance(manifest, (str, Path)):
        manifest = Path(manifest)
        if not manifest.is_file():
            raise FileNotFoundError(f"manifest missing: {manifest}")
    train_loader = loader_from_config(manifest, cfg, "train", shuffle=True)
    val_ds = MedicalDataset(manifest, image_size=cfg["data"]["image_size"], split="val", training=False,
                            roi_policy=cfg["data"].get("roi_policy", "none"),
                            quality_filter=cfg["data"].get("quality_filter", False),
                            min_quality=cfg["data"].get("min_quality", .35))
    device = resolve_device(cfg["training"].get("device", "auto"))
    model = (ClassicalRetrievalBaseline(cfg) if cfg["model"].get("kind") == "baseline" else QMedImageRetrievalAI(cfg)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"].get("learning_rate", 1e-4)),
                                 weight_decay=float(cfg["training"].get("weight_decay", 1e-5)))
    start_epoch, best_score, stale, history = 0, float("-inf"), 0, []
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = float(checkpoint.get("best_mAP", best_score))
    max_epochs = int(cfg["training"].get("epochs", 100))
    patience = int(cfg["training"].get("patience", 10))
    k = int(cfg["training"].get("monitor_k", 10))
    max_batches = int(cfg["training"].get("max_batches_per_epoch", 0))
    best_path = out / "best.pt"
    for epoch in range(start_epoch, max_epochs):
        model.train()
        t0 = time.perf_counter()
        epoch_loss, n_steps = 0.0, 0
        sums = {key: 0.0 for key in ("alignment", "contrastive", "triplet", "regularization")}
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            modality = batch["modality"].to(device, non_blocking=True)
            label = batch["uccs"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            z = model(image, modality)
            loss, parts = retrieval_losses(z, label, modality, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at epoch={epoch}, step={n_steps}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip_norm", 1.0)))
            optimizer.step()
            n_steps += 1
            epoch_loss += float(loss.detach())
            for key in sums:
                sums[key] += parts[key]
            if max_batches and n_steps >= max_batches:
                break
        if n_steps == 0:
            raise RuntimeError("zero train steps")
        vectors, meta = extract_embeddings(model, val_ds, device, cfg["data"]["batch_size"], cfg["data"].get("workers", 0))
        result, _ = evaluate_embeddings(vectors, meta, k_values=[k],
                         max_queries=int(cfg["training"].get("val_max_queries", 0)), seed=seed)
        map_score = float(result["mAP"])
        if not np.isfinite(map_score):
            map_score = -1.0
        row = {"epoch": epoch+1, "train_loss": epoch_loss/n_steps, "val_mAP": map_score,
               "val_eligible_queries": int(result["n_queries"]), "seconds": time.perf_counter()-t0}
        row.update({"loss_" + key: value/n_steps for key,value in sums.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        payload = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                   "config": cfg, "best_mAP": max(best_score,map_score), "seed": seed,
                   "val_result": result}
        torch.save(payload, out / "last.pt")
        if map_score > best_score + 1e-9:
            best_score = map_score
            stale = 0
            torch.save(payload, best_path)
        else:
            stale += 1
        if verbose:
            print(f"Epoch {epoch+1:03d}: loss={row['train_loss']:.4f} val_mAP={map_score:.4f} "
                  f"eligible={int(result['n_queries'])} time={row['seconds']:.1f}s", flush=True)
        if stale >= patience:
            if verbose:
                print(f"Early stopping after {stale} epochs without validation improvement")
            break
    if not best_path.is_file() and resume:
        best_path = Path(resume)
    write_json(out / "training_summary.json", {"best_mAP": best_score, "epochs_completed":len(history),
                                                "best_checkpoint": str(best_path), "train_images": len(train_loader.dataset),
                                                "val_images": len(val_ds), "seed": seed})
    return best_path


def load_checkpoint(checkpoint_path, device="cpu"):
    """Load a local trusted checkpoint, preserving the original architecture configuration."""
    device = resolve_device(str(device))
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload["config"]
    model = (ClassicalRetrievalBaseline(config) if config["model"].get("kind") == "baseline" else QMedImageRetrievalAI(config)).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, payload
