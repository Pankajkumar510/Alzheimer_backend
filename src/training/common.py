import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import WeightedRandomSampler
import pandas as pd


def seed_everything(seed: int = 42):
    import random
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, mixup_alpha: float = 0.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    import torch.nn.functional as F
    import random
    def mixup_data(x, y, alpha=0.2):
        if alpha <= 0:
            return x, y, y, 1.0
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                if mixup_alpha and mixup_alpha > 0.0:
                    X, y_a, y_b, lam = mixup_data(X, y, mixup_alpha)
                    logits = model(X)
                    loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                else:
                    logits = model(X)
                    loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if mixup_alpha and mixup_alpha > 0.0:
                X, y_a, y_b, lam = mixup_data(X, y, mixup_alpha)
                logits = model(X)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                logits = model(X)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


def validate(model, loader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    ys = []
    ps = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            running_loss += loss.item() * X.size(0)
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(logits.softmax(dim=1).cpu().numpy().tolist())
    ys = np.array(ys)
    ps = np.array(ps)
    preds = ps.argmax(axis=1)
    avg_loss = running_loss / max(len(ys), 1)
    num_classes = len(class_names)
    labels = list(range(num_classes))
    report = classification_report(ys, preds, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(ys, preds, labels=labels).tolist()
    return avg_loss, report, cm


def save_json(obj: Dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def compute_class_weights(train_csv: str, classes_json: str):
    df = pd.read_csv(train_csv)
    with open(classes_json, 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)
    counts = {k: 0 for k in class_to_idx.keys()}
    for _, row in df.iterrows():
        lbl = str(row['label'])
        if lbl in counts:
            counts[lbl] += 1
    total = sum(counts.values())
    num_classes = len(counts)
    weights = np.zeros(num_classes, dtype=np.float32)
    for lbl, idx in class_to_idx.items():
        n_c = counts.get(lbl, 1)
        weights[idx] = total / (num_classes * max(n_c, 1))
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(train_csv: str, classes_json: str):
    df = pd.read_csv(train_csv)
    with open(classes_json, 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)
    counts = {k: 0 for k in class_to_idx.keys()}
    labels = []
    for _, row in df.iterrows():
        lbl = str(row['label'])
        labels.append(lbl)
        if lbl in counts:
            counts[lbl] += 1
    total = sum(counts.values())
    num_classes = len(counts)
    class_weight = {lbl: total / (num_classes * max(c, 1)) for lbl, c in counts.items()}
    sample_weights = [class_weight.get(lbl, 1.0) for lbl in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def make_weighted_sampler_for_dataset(dataset, classes_json: str):
    """Create a WeightedRandomSampler aligned to the provided dataset (supports Subset of ImageCSVDataset)."""
    # Resolve base dataset and indices
    from torch.utils.data import Subset
    if isinstance(dataset, Subset):
        base_ds = dataset.dataset
        indices = dataset.indices
    else:
        base_ds = dataset
        indices = list(range(len(dataset)))

    # base_ds must have a .df with 'label' column
    # Use a duck-typing approach: expect a .df with a 'label' column
    if not hasattr(base_ds, 'df'):
        # fallback to uniform weights
        return WeightedRandomSampler(weights=[1.0]*len(indices), num_samples=len(indices), replacement=True)

    with open(classes_json, 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)
    counts = {k: 0 for k in class_to_idx.keys()}
    labels = []
    for i in indices:
        lbl = str(base_ds.df.iloc[i]["label"])  # type: ignore[attr-defined]
        labels.append(lbl)
        if lbl in counts:
            counts[lbl] += 1
    total = sum(counts.values())
    num_classes = len(counts)
    class_weight = {lbl: total / (num_classes * max(c, 1)) for lbl, c in counts.items()}
    sample_weights = [class_weight.get(lbl, 1.0) for lbl in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
