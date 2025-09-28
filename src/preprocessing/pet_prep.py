import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from .utils import (
    ensure_dir,
    is_nifti_file,
    is_dicom_dir,
    is_image_file,
    load_nifti_volume,
    load_dicom_volume,
    volume_to_multislice_rgb,
    resize_image,
    save_png,
    replicate_to_rgb,
    guess_patient_id,
    zscore,
)


def process_input(input_path: Path, output_dir: Path, label: str, image_size: int) -> List[Dict]:
    rows: List[Dict] = []
    if input_path.is_dir():
        for p in input_path.rglob("*"):
            if p.is_file() and (is_nifti_file(str(p)) or is_image_file(str(p))):
                rows.extend(process_input(p, output_dir, label, image_size))
            elif p.is_dir() and is_dicom_dir(str(p)):
                rows.extend(process_input(p, output_dir, label, image_size))
        return rows

    if is_nifti_file(str(input_path)):
        vol = load_nifti_volume(str(input_path))
        vol = zscore(vol)
        rgb = volume_to_multislice_rgb(vol, axis=0)
        img = resize_image(rgb, (image_size, image_size))
        pid = guess_patient_id(input_path)
        out_name = f"pet_{pid}.png"
        out_path = output_dir / out_name
        save_png(img, str(out_path))
        rows.append({
            "patient_id": pid,
            "file_path": str(out_path).replace("\\", "/"),
            "label": label,
            "modality": "PET",
            "slice_index": -1,
        })
        return rows

    if input_path.is_dir() and is_dicom_dir(str(input_path)):
        vol = load_dicom_volume(str(input_path))
        vol = zscore(vol)
        rgb = volume_to_multislice_rgb(vol, axis=0)
        img = resize_image(rgb, (image_size, image_size))
        pid = guess_patient_id(input_path)
        out_name = f"pet_{pid}.png"
        out_path = output_dir / out_name
        save_png(img, str(out_path))
        rows.append({
            "patient_id": pid,
            "file_path": str(out_path).replace("\\", "/"),
            "label": label,
            "modality": "PET",
            "slice_index": -1,
        })
        return rows

    if is_image_file(str(input_path)):
        import numpy as np
        import cv2
        arr = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            arr = cv2.cvtColor(cv2.imread(str(input_path)), cv2.COLOR_BGR2RGB)  # fallback
            arr = (arr.astype("float32") / 255.0)
        else:
            arr = arr.astype("float32")
            arr = zscore(arr)
            arr = replicate_to_rgb(arr)
        img = resize_image(arr, (image_size, image_size))
        pid = guess_patient_id(input_path)
        out_name = f"pet_{pid}.png"
        out_path = output_dir / out_name
        save_png(img, str(out_path))
        rows.append({
            "patient_id": pid,
            "file_path": str(out_path).replace("\\", "/"),
            "label": label,
            "modality": "PET",
            "slice_index": -1,
        })
        return rows

    return rows


def main():
    ap = argparse.ArgumentParser(description="PET preprocessing to 224x224 PNGs and CSV metadata")
    ap.add_argument("--input", required=True, help="Path to PET input (file or directory)")
    ap.add_argument("--output-dir", required=True, help="Directory to save processed PNGs")
    ap.add_argument("--metadata-out", required=True, help="Path to write metadata CSV")
    ap.add_argument("--label", default="Unknown", help="Label to assign if none available")
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(str(output_dir))

    rows = process_input(input_path, output_dir, args.label, args.image_size)
    if not rows:
        print("No PET inputs processed. Check paths and formats.")
        return
    df = pd.DataFrame(rows)
    Path(args.metadata_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.metadata_out, index=False)
    print(f"Wrote {len(df)} rows to {args.metadata_out}")


if __name__ == "__main__":
    main()
