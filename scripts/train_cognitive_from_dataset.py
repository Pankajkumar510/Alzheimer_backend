#!/usr/bin/env python3
"""
Train a cognitive MLP using the dataset in ../cognitive_dataset and save under backend/experiments.
Run: python scripts/train_cognitive_from_dataset.py
"""

import subprocess
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE.parent / "cognitive_dataset"
BACKEND = HERE
TRAIN = BACKEND / "src" / "training" / "train_mlp.py"
EXPERIMENTS = BACKEND / "experiments"

train_csv = DATA / "cognitive_train.csv"
val_csv = DATA / "cognitive_val.csv"
features_json = DATA / "cognitive_features.json"

if not train_csv.exists() or not val_csv.exists() or not features_json.exists():
    raise SystemExit(f"Dataset files not found in {DATA}. Expected cognitive_train.csv, cognitive_val.csv, cognitive_features.json")

cmd = [
    "python", str(TRAIN),
    "--train-csv", str(train_csv),
    "--val-csv", str(val_csv),
    "--features-json", str(features_json),
    "--epochs", "30",
    "--batch-size", "64",
    "--lr", "1e-3",
    "--out-dir", str(EXPERIMENTS)
]

# Set PYTHONPATH to include the backend directory
env = os.environ.copy()
env['PYTHONPATH'] = str(BACKEND)

print("Running:", " ".join(cmd))
subprocess.check_call(cmd, env=env)
print("Done. Cognitive model saved under:", EXPERIMENTS)
