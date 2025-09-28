import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_files(root: Path) -> List[Tuple[str, str]]:
    rows = []
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = cls_dir.name
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                rows.append((str(p), label))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Create train/val CSVs and classes.json from folder structure")
    ap.add_argument("--root", required=True, help="Root folder containing train/ and val/ (or test/) subfolders")
    ap.add_argument("--train-subdir", default="train")
    ap.add_argument("--val-subdir", default="test")
    ap.add_argument("--out-prefix", required=True, help="Prefix for output files (e.g., data/meta/mri)")
    args = ap.parse_args()

    root = Path(args.root)
    train_dir = root / args.train_subdir
    val_dir = root / args.val_subdir

    if not train_dir.exists() and not val_dir.exists():
        raise SystemExit(f"Neither train nor val directory exists under {root}")

    train_rows = collect_files(train_dir) if train_dir.exists() else []
    val_rows = collect_files(val_dir) if val_dir.exists() else []

    labels = sorted({lbl for _, lbl in train_rows + val_rows})
    class_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    def to_df(rows):
        return pd.DataFrame([
            {"file_path": path, "label": label}
            for path, label in rows
        ])

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if train_rows:
        df_tr = to_df(train_rows)
        df_tr.to_csv(str(out_prefix) + "_train.csv", index=False)
    if val_rows:
        df_va = to_df(val_rows)
        df_va.to_csv(str(out_prefix) + "_val.csv", index=False)

    with open(str(out_prefix) + "_classes.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"Train samples: {len(train_rows)} | Val samples: {len(val_rows)} | Classes: {len(labels)}")
    print(f"Wrote {str(out_prefix)}_train.csv, {str(out_prefix)}_val.csv, {str(out_prefix)}_classes.json")


if __name__ == "__main__":
    main()
