"""Image-only preprocessing; annotation-assisted ROIs are an explicit opt-in."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


def _read_nifti(path: str, slice_idx: int) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as error:
        raise ImportError("NIfTI requires: pip install -e '.[medical]'") from error
    volume = _cached_nib(path)
    if slice_idx < 0 or slice_idx >= volume.shape[-1]:
        raise IndexError(f"slice {slice_idx} outside {path}, shape={volume.shape}")
    return np.asarray(volume.dataobj[..., slice_idx], dtype=np.float32)


@lru_cache(maxsize=12)
def _cached_nib(path: str):
    import nibabel as nib
    return nib.load(path)


def read_grayscale(path: str, slice_idx: int = 0) -> np.ndarray:
    path = str(path)
    lower = path.lower()
    if lower.endswith((".nii", ".nii.gz")):
        return _read_nifti(path, slice_idx)
    if lower.endswith(".npy"):
        a = np.load(path, allow_pickle=False)
        if a.ndim == 3:
            a = a[..., slice_idx]
        return a.astype(np.float32)
    if lower.endswith(".dcm"):
        try:
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
        except ImportError as error:
            raise ImportError("DICOM requires: pip install -e '.[medical]'") from error
        ds = pydicom.dcmread(path)
        pixels = ds.pixel_array
        pixels = apply_voi_lut(apply_modality_lut(pixels, ds), ds).astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = float(np.max(pixels)) - pixels
        return pixels
    with Image.open(path) as img:
        return np.array(img.convert("L"), dtype=np.float32)


def normalize_image(arr: np.ndarray, image_size: int) -> torch.Tensor:
    """Min-max normalisation with robust handling of empty and nonfinite images."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D gray image, got shape {arr.shape}")
    finite = np.isfinite(arr)
    if not np.any(finite):
        arr = np.zeros_like(arr)
    else:
        low, high = float(np.min(arr[finite])), float(np.max(arr[finite]))
        arr = np.nan_to_num(arr, nan=low, posinf=high, neginf=low)
        if high > low:
            arr = (arr - low) / (high - low)
        else:
            arr = np.zeros_like(arr)
    tensor = torch.from_numpy(arr.copy())[None]
    tensor = torch.nn.functional.interpolate(tensor[None], size=(image_size, image_size), mode="bilinear", align_corners=False)[0]
    return tensor.clamp(0.0, 1.0)


def quality_score(image: torch.Tensor) -> float:
    """Paper's three-component sharpness/contrast/signal proxy, monotonically normalized."""
    x = image[0].detach().cpu().numpy()
    from scipy.ndimage import laplace
    sharpness = float(np.var(laplace(x)))
    sharp = sharpness / (sharpness + 0.01)
    rms = float(np.std(x))
    contrast = rms / (rms + 0.15)
    intact = float(np.mean((x > 1e-3) & (x < 1.0 - 1e-3)))
    return (sharp + contrast + intact) / 3.0


def augment(image: torch.Tensor) -> torch.Tensor:
    """One spatial transform shared across all MRI sequences; train only."""
    import random
    angle = random.uniform(-10, 10)
    translate = [int(random.uniform(-.05, .05) * image.shape[-1]),
                 int(random.uniform(-.05, .05) * image.shape[-2])]
    scale = random.uniform(.9, 1.1)
    image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=[0., 0.], interpolation=InterpolationMode.BILINEAR)
    if random.random() < .5:
        image = TF.hflip(image)
    image = image * random.uniform(.9, 1.1)
    # torchvision.adjust_brightness/contrast only supports 1 or 3 channels;
    # operate directly on all four aligned MRI sequence channels instead.
    mean = image.mean(dim=(-2, -1), keepdim=True)
    image = (image - mean) * random.uniform(.9, 1.1) + mean
    return image.clamp(0, 1)


def apply_roi(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[-2:] != image.shape[-2:]:
        mask = torch.nn.functional.interpolate(mask[None], image.shape[-2:], mode="nearest")[0]
    return image * (.25 + .75 * (mask > .5).float())
