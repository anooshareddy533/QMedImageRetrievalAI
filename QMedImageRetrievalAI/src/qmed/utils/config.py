"""Configuration, provenance and reproducibility helpers."""
from __future__ import annotations
import json
import os
import random
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import yaml


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    with open(path, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if overrides:
        for key, value in overrides.items():
            cursor = cfg
            parts = key.split(".")
            for part in parts[:-1]:
                cursor = cursor[part]
            cursor[parts[-1]] = value
    return cfg


def resolve_device(option: str = "auto") -> torch.device:
    if option == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(option).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; set training.device=cpu")
    return torch.device(option)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def write_json(path: str | Path, data: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def copy_config(config):
    return deepcopy(config)
