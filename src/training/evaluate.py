import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.datasets import ImageCSVDataset
from src.models.convnext_mri import create_mri_model, get_transforms as mri_tx
from src.models.vit_pet import create_vit_model, get_transforms as pet_tx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--classes-json", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model-type", choices=["mri", "pet"], required=True)
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    with open(args.classes_json, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    transform = mri_tx(args.image_size, False) if args.model_type == "mri" else pet_tx(args.image_size, False)
    ds = ImageCSVDataset(args.csv, args.classes_json, transform=transform)
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "mri":
        model = create_mri_model(num_classes=num_classes)
    else:
        model = create_vit_model(num_classes=num_classes)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state"])  # type: ignore
    model.to(device)
    model.eval()

    ys = []
    ps = []
    with torch.no_grad():
        for X, y in dl:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(logits.softmax(dim=1).cpu().numpy().tolist())
    ys = np.array(ys)
    ps = np.array(ps)
    preds = ps.argmax(axis=1)

    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(ys, preds, target_names=[idx_to_class[i] for i in range(num_classes)], output_dict=True)
    cm = confusion_matrix(ys, preds).tolist()
    result = {"classification_report": report, "confusion_matrix": cm}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
