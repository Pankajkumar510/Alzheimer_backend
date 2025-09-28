import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse
import os
import tempfile
import shutil
import uuid

import numpy as _np
from PIL import Image as _Image
try:
    import pydicom as _pyd
except Exception:  # pragma: no cover
    _pyd = None
try:
    import nibabel as _nib
except Exception:  # pragma: no cover
    _nib = None


def _brain_like_score(img_path: str) -> float:
    """Return a heuristic score in [0,1] for whether an image looks like a brain scan.
    Uses symmetry, aspect, center/border contrast, edge patterns, and colorfulness (if present).
    Research-only, lightweight, and conservative.
    """
    try:
        return _brain_like_details(img_path)["score"]
    except Exception:
        return 0.0


def _brain_like_details(img_path: str) -> dict:
    """Return detailed metrics for screening.
    Keys: score, sym, aspect, contrast, edge_ratio, color_penalty, border_dark_frac, center_bright_frac, w, h
    """
    img_rgb = _Image.open(img_path).convert("RGB")
    w, h = img_rgb.size
    rgb = _np.array(img_rgb, dtype=_np.float32) / 255.0
    R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
    rg = R - G
    yb = 0.5*(R + G) - B
    c_var = _np.sqrt(rg.var() + yb.var())
    gray_dev = (abs(R-G).mean() + abs(G-B).mean())/2.0
    color_penalty = 0.0
    if c_var > 0.25 and gray_dev > 0.08:
        color_penalty = 0.25
    arr = _np.dot(rgb, _np.array([0.2989,0.5870,0.1140], dtype=_np.float32))
    arr_flipped = _np.flip(arr, axis=1)
    a = arr - arr.mean()
    b = arr_flipped - arr_flipped.mean()
    denom = (a.std() * b.std()) + 1e-6
    corr = float((a * b).mean() / denom)
    sym = max(0.0, min(1.0, (corr + 1.0) / 2.0))
    ratio = max(w, h) / max(1.0, min(w, h))
    aspect = 1.0 - max(0.0, min(1.0, (ratio - 1.0) / 0.3))
    yy, xx = _np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    r = _np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    R = min(w, h) / 2.0
    center_mask = r <= (0.35 * R)
    outer_mask = (r >= (0.45 * R)) & (r <= (0.5 * R))
    # Outer border ring (very outer) for darkness check
    outer2_mask = r >= (0.80 * R)
    cmean = float(arr[center_mask].mean()) if center_mask.any() else 0.0
    omean = float(arr[outer_mask].mean()) if outer_mask.any() else 0.0
    diff = cmean - omean
    contrast = max(0.0, min(1.0, (diff + 0.2) / 0.6))
    # Darkness at border, brightness in center
    border_dark_frac = float((arr[outer2_mask] < 0.12).mean()) if outer2_mask.any() else 0.0
    center_bright_frac = float((arr[center_mask] > 0.25).mean()) if center_mask.any() else 0.0
    dx = _np.zeros_like(arr)
    dy = _np.zeros_like(arr)
    dx[:,1:] = arr[:,1:] - arr[:,:-1]
    dy[1:,:] = arr[1:,:] - arr[:-1,:]
    mag = _np.sqrt(dx*dx + dy*dy)
    corner_mask = _np.zeros_like(arr, dtype=bool)
    bsz = max(1, int(0.15*min(h,w)))
    corner_mask[:bsz,:bsz] = True
    corner_mask[:bsz,-bsz:] = True
    corner_mask[-bsz:,:bsz] = True
    corner_mask[-bsz:,-bsz:] = True
    center_mag = float(mag[center_mask].mean() if center_mask.any() else 0.0)
    corner_mag = float(mag[corner_mask].mean() if corner_mask.any() else 0.0)
    edge_ratio = max(0.0, min(1.0, (center_mag - corner_mag + 0.05) / 0.3))
    score = 0.40 * sym + 0.10 * aspect + 0.25 * contrast + 0.15 * edge_ratio + 0.10 * max(0.0, min(1.0, (border_dark_frac - 0.4) / 0.4))
    score = max(0.0, min(1.0, score - color_penalty))
    return {
        "score": float(score),
        "sym": float(sym),
        "aspect": float(aspect),
        "contrast": float(contrast),
        "edge_ratio": float(edge_ratio),
        "color_penalty": float(color_penalty),
        "border_dark_frac": float(border_dark_frac),
        "center_bright_frac": float(center_bright_frac),
        "w": int(w),
        "h": int(h),
    }

from src.inference.predict import InferenceEngine
from src.api.schemas import PredictResponse


BASE_DIR = Path(__file__).resolve().parents[2]
router = APIRouter()


@router.get("/")
async def index():
    return {
        "status": "ok",
        "message": "Welcome to the Multimodal Alzheimer's Detection API",
        "docs": "/docs",
        "ui": "/ui",
        "endpoints": {
            "GET /health": "service health check",
            "GET /predict": "usage help for the prediction endpoint",
            "POST /predict": "submit assessment via multipart/form-data",
            "GET /report/{patient_id}": "generate and return a PDF report path"
        }
    }


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/predict")
async def predict_help():
    return {
        "status": "ready",
        "detail": "Use POST multipart/form-data to /predict to run inference.",
        "fields": {
            "patient_id": "string (required)",
            "mri_file": "file upload (optional)",
            "pet_file": "file upload (optional)",
            "cognitive": "JSON string (optional). Example: {\"mmse\": 28, \"age\": 73}"
        },
        "example_curl": "curl -X POST http://localhost:8000/predict -F 'patient_id=demo' -F 'mri_file=@/path/to/mri.jpg' -F 'pet_file=@/path/to/pet.jpg' -F 'cognitive={\\\"mmse\\\":28}'"
    }


@router.post("/predict", response_model=PredictResponse)
async def predict(
    patient_id: str = Form(...),
    mri_file: Optional[UploadFile] = File(None),
    pet_file: Optional[UploadFile] = File(None),
    cognitive: Optional[str] = Form(None),
):
    uploads_dir = BASE_DIR / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    conv_dir = uploads_dir / "converted"
    conv_dir.mkdir(parents=True, exist_ok=True)

    async def _save_upload(ufile: UploadFile, prefix: str) -> str:
        target = uploads_dir / f"{patient_id}_{prefix}_{ufile.filename}"
        with open(target, "wb") as f:
            f.write(await ufile.read())
        return str(target)

    async def _maybe_convert_for_infer(path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        p = Path(path)
        ext = p.suffix.lower()
        name = p.stem
        try:
            if ext in {".jpg", ".jpeg", ".png"}:
                return path
            if ext == ".dcm" and _pyd is not None:
                ds = _pyd.dcmread(str(p))
                arr = ds.pixel_array.astype("float32")
                # normalize
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
                arr = (arr * 255.0).astype("uint8")
                img = _Image.fromarray(arr)
                out = conv_dir / f"{name}.png"
                img.save(str(out))
                return str(out)
            if ext in {".nii", ".gz"} and _nib is not None:
                img = _nib.load(str(p))
                data = img.get_fdata()
                # choose central slice along the largest axis
                axis = int(_np.argmax(data.shape))
                slicer = [slice(None)] * len(data.shape)
                slicer[axis] = data.shape[axis] // 2
                arr = data[tuple(slicer)]
                arr = _np.nan_to_num(arr).astype("float32")
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
                arr = (arr * 255.0).astype("uint8")
                img2d = _Image.fromarray(arr)
                out = conv_dir / f"{name}.png"
                img2d.save(str(out))
                return str(out)
        except Exception:
            # return original if conversion fails; inference may skip
            return path
        return path

    mri_path = None
    pet_path = None

    if mri_file is not None:
        mri_path = await _save_upload(mri_file, "mri")
        mri_path = await _maybe_convert_for_infer(mri_path)
    if pet_file is not None:
        pet_path = await _save_upload(pet_file, "pet")
        pet_path = await _maybe_convert_for_infer(pet_path)

    # Basic guard against random/non-brain images (research-only heuristic)
    screen = {"threshold": 0.6}
    if mri_path:
        try:
            det = _brain_like_details(mri_path)
        except Exception as e:
            det = {"score": 0.0, "sym": 0.0, "aspect": 0.0, "contrast": 0.0, "edge_ratio": 0.0, "color_penalty": 0.0, "border_dark_frac": 0.0, "center_bright_frac": 0.0, "error": str(e)}
            screen["mri_error"] = str(e)
        screen["mri"] = det
    if pet_path:
        try:
            detp = _brain_like_details(pet_path)
        except Exception as e:
            detp = {"score": 0.0, "sym": 0.0, "aspect": 0.0, "contrast": 0.0, "edge_ratio": 0.0, "color_penalty": 0.0, "border_dark_frac": 0.0, "center_bright_frac": 0.0, "error": str(e)}
            screen["pet_error"] = str(e)
        screen["pet"] = detp
    def _passes_mri(det: dict) -> bool:
        if det["score"] >= screen["threshold"]:
            return True
        # Dynamic relaxation for typical grayscale MRI-like images: require dark border and some center brightness
        if (
            det["score"] >= 0.50 and det["color_penalty"] == 0.0 and det["sym"] >= 0.35 and det["aspect"] >= 0.6 and det["edge_ratio"] >= 0.25
            and det.get("border_dark_frac", 0.0) >= 0.55 and det.get("center_bright_frac", 0.0) >= 0.25
        ):
            return True
        return False

    def _passes_pet(det: dict) -> bool:
        if det["score"] >= screen["threshold"]:
            return True
        # PET images may be less symmetric; relax symmetry but require structure
        if det["score"] >= 0.55 and det["edge_ratio"] >= 0.30 and det["aspect"] >= 0.6:
            return True
        return False
    fail = False
    if mri_path:
        fail = fail or (not _passes_mri(screen["mri"]))
    if pet_path:
        fail = fail or (not _passes_pet(screen["pet"]))
    if fail:
        return {
            "status": "error",
            "result": {
                "patient_id": patient_id,
                "inputs": {"mri_path": mri_path, "pet_path": pet_path, "cognitive": cognitive_dict},
                "predictions": {},
                "error": f"Uploaded image(s) do not appear to be brain scans (screen={screen}). Please upload valid MRI/PET brain images.",
            },
        }

    cognitive_dict = None
    if cognitive:
        try:
            cognitive_dict = json.loads(cognitive)
        except Exception:
            cognitive_dict = None

    mri_list = list((BASE_DIR / "experiments").glob("mri_*/best_model.pth"))
    pet_list = list((BASE_DIR / "experiments").glob("pet_*/best_model.pth"))
    cog_list = list((BASE_DIR / "experiments").glob("cognitive_*"))
    engine = InferenceEngine(
        mri_classes_json=str(BASE_DIR / "data" / "meta" / "mri_classes.json"),
        pet_classes_json=str(BASE_DIR / "data" / "meta" / "pet_classes.json"),
        mri_weights=str(mri_list[-1]) if mri_list else None,
        pet_weights=str(pet_list[-1]) if pet_list else None,
        cognitive_dir=str(cog_list[-1]) if cog_list else None,
        fusion_mapping_json=str(BASE_DIR / "data" / "meta" / "fusion_mapping.json"),
    )

    try:
        result = engine.predict(
            patient_id=patient_id,
            mri_path=mri_path,
            pet_path=pet_path,
            cognitive=cognitive_dict,
            save_explain_dir=str(BASE_DIR / "static" / "explain"),
        )
    except Exception as e:
        # Never hard-fail the request; return a structured error
        return {
            "status": "error",
            "result": {
                "patient_id": patient_id,
                "inputs": {"mri_path": mri_path, "pet_path": pet_path, "cognitive": cognitive_dict},
                "predictions": {},
                "error": str(e),
            },
        }

    return {"status": "ok", "result": result}


@router.get("/report/{patient_id}")
async def report(patient_id: str):
    # minimal PDF report using reportlab; stream file back to client
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    out_dir = BASE_DIR / "experiments" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patient_id}.pdf"

    c = canvas.Canvas(str(out_path), pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 14)
    c.drawString(72, height - 72, f"Patient Report: {patient_id}")
    c.setFont("Helvetica", 10)
    c.drawString(72, height - 100, "Generated by the Multimodal Alzheimer's Detection System (research-only)")
    c.showPage()
    c.save()

    return FileResponse(path=str(out_path), media_type="application/pdf", filename=f"report_{patient_id}.pdf")
