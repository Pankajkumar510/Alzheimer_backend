import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, Request
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
            "POST /validate-image": "validate if an image appears to be a brain scan",
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


    # Check for duplicate files being uploaded as different types
    if mri_file and pet_file and mri_file.filename == pet_file.filename:
        print(f"⚠️  WARNING: Same file uploaded as both MRI and PET: {mri_file.filename}")
    elif mri_file and not pet_file:
        print(f"ℹ️  INFO: Only MRI file uploaded")
    elif pet_file and not mri_file:
        print(f"ℹ️  INFO: Only PET file uploaded")
    # Parse cognitive JSON early so it is available even on early-return paths
    def _parse_cognitive(val: Optional[str]) -> Optional[dict]:
        if val is None:
            return None
        t = str(val).strip()
        if not t:
            return None
        try:
            return json.loads(t)
        except Exception:
            # tolerate single-quoted keys/strings and loose JSON from some clients
            try:
                t2 = t.replace("'", '"')
                return json.loads(t2)
            except Exception:
                # As a last resort, treat presence of a value as a request for cognitive-only inference
                # with default answers (empty dict). This ensures the cognitive-only path still runs.
                return {}

    cognitive_dict = _parse_cognitive(cognitive)

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
    screen = {"threshold": 0.45}  # Balanced threshold for brain detection
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
    
    failed_files = []
    validation_details = {}
    
    if mri_path:
        if not _passes_mri(screen["mri"], screen["threshold"]):
            failed_files.append("MRI")
            validation_details["mri"] = screen["mri"]
    if pet_path:
        if not _passes_pet(screen["pet"], screen["threshold"]):
            failed_files.append("PET")
            validation_details["pet"] = screen["pet"]
    
    if failed_files:
        # Return error response for invalid medical images
        return {
            "status": "error",
            "result": {
                "error": "Invalid medical image detected",
                "message": f"The uploaded {' and '.join(failed_files).lower()} image(s) do not appear to be valid brain scans. Please upload authentic MRI or PET scan images.",
                "details": {
                    "failed_validation": failed_files,
                    "validation_scores": validation_details,
                    "requirements": {
                        "MRI": "Should be a grayscale brain scan image with dark borders and bright center regions, showing typical brain anatomy",
                        "PET": "Should be a brain metabolism scan with clear anatomical structure and appropriate contrast"
                    },
                    "supported_formats": ["JPG", "PNG", "DICOM (.dcm)", "NIfTI (.nii, .nii.gz)"]
                }
            }
        }

    # Use the specific trained models
    mri_model_path = BASE_DIR.parent / "mri_20251002-101653" / "mri_best_model.pth"
    pet_model_path = BASE_DIR.parent / "pet_20251002-022237" / "pet_best_model.pth"
    fusion_mapping_path = BASE_DIR.parent / "fusion_mapping.json"
    
    # Look for cognitive models in experiments
    cog_list = list((BASE_DIR / "experiments").glob("cognitive_*"))
    engine = InferenceEngine(
        mri_classes_json=str(BASE_DIR / "meta" / "mri_classes.json"),
        pet_classes_json=str(BASE_DIR / "meta" / "pet_classes.json"),
        mri_weights=str(mri_model_path) if mri_model_path.exists() else None,
        pet_weights=str(pet_model_path) if pet_model_path.exists() else None,
        cognitive_dir=str(cog_list[-1]) if cog_list else None,
        fusion_mapping_json=str(fusion_mapping_path) if fusion_mapping_path.exists() else None,
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

    response_data = {"status": "ok", "result": result}
    print(f"=== PREDICT RESPONSE SENT ===")
    print(f"Status: {response_data['status']}")
    if 'predictions' in result:
        print(f"Predictions sent: {list(result['predictions'].keys())}")
        for k, v in result['predictions'].items():
            if 'probabilities' in v:
                conf = max(v['probabilities']) * 100
                print(f"  {k}: {v.get('label', 'N/A')} ({conf:.1f}%)")
    print(f"================================\n")
    return response_data


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


@router.post("/predict2")
async def predict2(request: Request):
    # Manual form parsing fallback endpoint
    form = await request.form()
    patient_id = str(form.get('patient_id') or f"patient_{uuid.uuid4().hex[:8]}")
    mri_file = form.get('mri_file')
    pet_file = form.get('pet_file')
    cognitive = form.get('cognitive')
    # Coerce filenames
    try:
        mri_name = getattr(mri_file, 'filename', None)
        pet_name = getattr(pet_file, 'filename', None)
    except Exception:
        mri_name = pet_name = None
    # Build UploadFile-like objects are already UploadFile
    # Reuse predict coroutine by calling its internals via direct import of this module function
    # We'll call the same logic as predict by delegating to InferenceEngine
    # Parse cognitive
    def _parse(val):
        if not val:
            return None
        try:
            return json.loads(str(val))
        except Exception:
            try:
                return json.loads(str(val).replace("'", '"'))
            except Exception:
                return {}
    cognitive_dict = _parse(cognitive)
    # Save files if present
    uploads_dir = BASE_DIR / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    async def _save(uf, prefix):
        if uf is None:
            return None
        path = uploads_dir / f"{patient_id}_{prefix}_{uf.filename}"
        with open(path, 'wb') as f:
            f.write(await uf.read())
        return str(path)
    mri_path = await _save(mri_file, 'mri') if mri_file else None
    pet_path = await _save(pet_file, 'pet') if pet_file else None
    # Engine as in predict
    mri_model_path = BASE_DIR.parent / "mri_20251002-101653" / "mri_best_model.pth"
    pet_model_path = BASE_DIR.parent / "pet_20251002-022237" / "pet_best_model.pth"
    fusion_mapping_path = BASE_DIR.parent / "fusion_mapping.json"
    cog_list = list((BASE_DIR / "experiments").glob("cognitive_*"))
    engine = InferenceEngine(
        mri_classes_json=str(BASE_DIR / "meta" / "mri_classes.json"),
        pet_classes_json=str(BASE_DIR / "meta" / "pet_classes.json"),
        mri_weights=str(mri_model_path) if mri_model_path.exists() else None,
        pet_weights=str(pet_model_path) if pet_model_path.exists() else None,
        cognitive_dir=str(cog_list[-1]) if cog_list else None,
        fusion_mapping_json=str(fusion_mapping_path) if fusion_mapping_path.exists() else None,
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
        return {"status": "error", "result": {"error": str(e)}}
    return {"status":"ok","result": result}

@router.post("/validate-image")
async def validate_image(
    image_file: UploadFile = File(...),
    scan_type: str = Form(...)  # "mri" or "pet"
):
    """Validate if an uploaded image appears to be a valid brain scan.
    Used by frontend for pre-upload validation."""
    
    # Save temporary file for validation
    uploads_dir = BASE_DIR / "data" / "uploads" / "temp"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    temp_path = uploads_dir / f"temp_{image_file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await image_file.read())
        
        # Convert if needed (same logic as in predict)
        converted_path = await _convert_for_validation(str(temp_path))
        
        # Run brain screening
        try:
            details = _brain_like_details(converted_path)
        except Exception as e:
            return {
                "valid": False,
                "reason": "Failed to process image",
                "details": {"error": str(e)}
            }
        
        # Apply validation logic based on scan type
        threshold = 0.45  # Balanced threshold for real medical images
        is_valid = False
        
        if scan_type.lower() == "mri":
            is_valid = _passes_mri(details, threshold)
        elif scan_type.lower() == "pet":
            is_valid = _passes_pet(details, threshold)
        
        response = {
            "valid": is_valid,
            "score": details["score"],
            "scan_type": scan_type,
            "filename": image_file.filename,
            "details": {
                "symmetry": details["sym"],
                "aspect_ratio": details["aspect"], 
                "contrast": details["contrast"],
                "edge_ratio": details["edge_ratio"],
                "color_penalty": details["color_penalty"]
            }
        }
        
        if not is_valid:
            response["reason"] = f"Image does not appear to be a valid {scan_type.upper()} brain scan"
            response["requirements"] = {
                "MRI": "Should be a grayscale brain scan with dark borders, bright center regions, and bilateral symmetry",
                "PET": "Should show brain metabolism with clear anatomical structure and appropriate contrast"
            }.get(scan_type.upper(), "Should be a valid brain scan image")
        
        return response
        
    finally:
        # Clean up temporary file
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _passes_mri(details: dict, threshold: float = 0.45) -> bool:
    """Enhanced MRI validation logic - more lenient for real medical images."""
    
    # First, reject obviously invalid images (high color penalty = colorful/random images)
    if details["color_penalty"] >= 0.25:  # Reject colorful/random images (changed to >= to catch 0.25)
        return False
    
    # Primary threshold check - if score is good, accept with minimal checks
    if details["score"] >= threshold:
        # Basic sanity checks for high-scoring images
        if details["aspect"] < 0.50:  # Extremely non-square
            return False
        return True
    
    # Secondary validation for borderline cases (more flexible for real brain scans)
    # Real MRI scans might have lower symmetry due to anatomical variations or pathology
    if (
        details["score"] >= 0.25 and  # Lower minimum score for real scans
        details["aspect"] >= 0.55 and  # More flexible aspect ratio
        details["color_penalty"] <= 0.20  # Allow some color variation
    ):
        # At least one brain-like characteristic should be present
        if (
            details["contrast"] >= 0.12 or  # Some contrast
            details["sym"] >= 0.30 or  # Some symmetry
            details["edge_ratio"] >= 0.10 or  # Some edge structure
            details.get("center_bright_frac", 0.0) >= 0.10  # Some central brightness
        ):
            return True
    
    # Tertiary check - very permissive for challenging MRI scans
    if (
        details["score"] >= 0.20 and  # Very low minimum
        details["color_penalty"] <= 0.15 and  # Mostly grayscale
        details["aspect"] >= 0.60 and  # Reasonably square
        (
            # Any one of these indicates brain-like structure:
            details["contrast"] >= 0.18 or  # Good contrast (MRI typically has good contrast)
            details["sym"] >= 0.50 or  # Good symmetry
            details["edge_ratio"] >= 0.25 or  # Good edge definition
            (details.get("center_bright_frac", 0.0) >= 0.20 and details.get("border_dark_frac", 0.0) >= 0.40)
        )
    ):
        return True
    
    # Final fallback - for grayscale images with basic MRI characteristics
    if (
        details["score"] >= 0.15 and  # Absolute minimum
        details["color_penalty"] == 0.0 and  # Must be perfectly grayscale
        details["aspect"] >= 0.70 and  # Reasonably square
        details["contrast"] >= 0.15 and  # MRIs should have decent contrast
        (
            details.get("center_bright_frac", 0.0) >= 0.15 or  # Bright center (white matter)
            details.get("border_dark_frac", 0.0) >= 0.60 or  # Dark borders
            details["sym"] >= 0.65  # Good symmetry
        )
    ):
        return True
        
    return False


async def _convert_for_validation(path: str) -> str:
    """Convert DICOM/NIfTI files to PNG for validation, similar to _maybe_convert_for_infer."""
    p = Path(path)
    ext = p.suffix.lower()
    name = p.stem
    conv_dir = p.parent / "converted"
    conv_dir.mkdir(parents=True, exist_ok=True)
    
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
        # return original if conversion fails
        return path
    return path


def _passes_pet(details: dict, threshold: float = 0.45) -> bool:
    """Enhanced PET validation logic - more lenient for real medical images."""
    
    # First, reject obviously invalid images (high color penalty = colorful/random images)
    if details["color_penalty"] >= 0.25:  # Reject colorful/random images (changed to >= to catch 0.25)
        return False
    
    # Primary threshold check - if score is good, accept with minimal checks
    if details["score"] >= threshold:
        # Very basic sanity checks for high-scoring images
        if details["aspect"] < 0.50:  # Extremely non-square
            return False
        return True
    
    # Secondary validation - PET scans are often more challenging
    # Real PET scans may have lower symmetry, contrast, and edge definition
    if (
        details["score"] >= 0.25 and  # Lower baseline for real PET scans
        details["aspect"] >= 0.55 and  # More flexible aspect ratio
        details["color_penalty"] <= 0.20  # Allow some color variation (artifacts, compression)
    ):
        # At least one of these characteristics should be present:
        if (
            details["contrast"] >= 0.10 or  # Some contrast (very low threshold)
            details["edge_ratio"] >= 0.12 or  # Some edge structure
            details["sym"] >= 0.30 or  # Some symmetry
            details.get("center_bright_frac", 0.0) >= 0.05  # Some central activity
        ):
            return True
    
    # Tertiary check - very permissive for challenging PET scans
    # This handles edge cases like very low contrast or asymmetric pathological cases
    if (
        details["score"] >= 0.20 and  # Very low minimum score
        details["color_penalty"] <= 0.15 and  # Mostly grayscale
        details["aspect"] >= 0.60 and  # Square-ish
        (
            # Any one of these indicates brain-like structure:
            details["contrast"] >= 0.15 or  # Reasonable contrast
            details["sym"] >= 0.50 or  # Good symmetry
            details["edge_ratio"] >= 0.20 or  # Good edge structure
            (details.get("border_dark_frac", 0.0) >= 0.60 and details["contrast"] >= 0.08)  # Dark borders + some contrast
        )
    ):
        return True
    
    # Final fallback - for grayscale images with minimal brain-like characteristics
    # This is very permissive to catch edge cases of real medical images
    if (
        details["score"] >= 0.15 and  # Absolute minimum
        details["color_penalty"] == 0.0 and  # Must be perfectly grayscale
        details["aspect"] >= 0.70 and  # Reasonably square
        details["contrast"] >= 0.10 and  # Minimal contrast
        (
            details.get("center_bright_frac", 0.0) >= 0.10 or
            details.get("border_dark_frac", 0.0) >= 0.70 or
            details["sym"] >= 0.60
        )
    ):
        return True
        
    return False
