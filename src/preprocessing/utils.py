import os
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import nibabel as nib
from PIL import Image

try:
    import pydicom
    from pydicom import dcmread
except Exception:  # pragma: no cover
    pydicom = None
    dcmread = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def is_nifti_file(path: str) -> bool:
    p = str(path).lower()
    return p.endswith(".nii") or p.endswith(".nii.gz")


def is_image_file(path: str) -> bool:
    p = str(path).lower()
    return any(p.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"])


def is_dicom_dir(path: str) -> bool:
    if pydicom is None:
        return False
    p = Path(path)
    if not p.is_dir():
        return False
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() == ".dcm":
            return True
    # sometimes dicom without extension
    for f in p.iterdir():
        if f.is_file():
            try:
                dcmread(str(f))
                return True
            except Exception:
                continue
    return False


def load_nifti_volume(path: str) -> np.ndarray:
    img = nib.load(str(path))
    vol = img.get_fdata().astype(np.float32)
    return vol


def load_dicom_volume(dir_path: str) -> np.ndarray:
    assert pydicom is not None, "pydicom not installed"
    files = []
    for f in Path(dir_path).iterdir():
        if f.is_file():
            try:
                ds = dcmread(str(f), stop_before_pixels=False)
                files.append(ds)
            except Exception:
                continue
    if not files:
        raise ValueError(f"No DICOM files found in {dir_path}")
    # sort by InstanceNumber if available
    def _key(ds):
        return getattr(ds, "InstanceNumber", 0)

    files.sort(key=_key)
    slices = []
    for ds in files:
        arr = ds.pixel_array.astype(np.float32)
        slices.append(arr)
    vol = np.stack(slices, axis=0)  # (Z, H, W)
    return vol


def zscore(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = float(np.mean(arr))
    s = float(np.std(arr))
    return (arr - m) / (s + eps)


def minmax(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    return (arr - mn) / (mx - mn + eps)


def center_index(length: int) -> int:
    return int(length // 2)


def volume_to_multislice_rgb(vol: np.ndarray, axis: int = 0, center: Optional[int] = None) -> np.ndarray:
    """
    Convert 3D volume to 3-channel RGB by taking center-1, center, center+1 slices
    along the specified axis. Pads with edge slices if out of bounds.
    Output dtype float32 in [0, 1].
    """
    if vol.ndim != 3:
        raise ValueError("volume must be 3D (Z,H,W)")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")

    shape = vol.shape
    if center is None:
        center = center_index(shape[axis])
    idxs = [max(center - 1, 0), center, min(center + 1, shape[axis] - 1)]

    def get_slice(i):
        if axis == 0:
            return vol[i, :, :]
        elif axis == 1:
            return vol[:, i, :]
        else:
            return vol[:, :, i]

    chs = [get_slice(i) for i in idxs]  # list of (H, W)
    chs = [minmax(c) for c in chs]
    rgb = np.stack(chs, axis=-1).astype(np.float32)  # (H, W, 3)
    return rgb


def resize_image(img: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    # img expected HxWxC or HxW
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    resized = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    resized = np.clip(resized, 0.0, 1.0).astype(np.float32)
    return resized


def save_png(img01: np.ndarray, out_path: str) -> None:
    # expects img in [0,1], HxWxC
    arr = (np.clip(img01, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def replicate_to_rgb(img2d: np.ndarray) -> np.ndarray:
    img01 = minmax(img2d)
    rgb = np.stack([img01] * 3, axis=-1).astype(np.float32)
    return rgb


def guess_patient_id(path: Path) -> str:
    stem = path.stem
    # try to extract patient id-ish tokens
    m = re.findall(r"[A-Za-z0-9]+", stem)
    if m:
        return "_".join(m)[:64]
    return uuid.uuid5(uuid.NAMESPACE_URL, str(path)).hex[:16]
