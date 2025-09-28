import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.models.mlp_cognitive import CognitiveMLP, fit_scaler, save_scaler
from src.training.common import seed_everything, save_json, timestamp


class TabularDataset(Dataset):
    def __init__(self, csv_path: str, features: list, target_col: str):
        self.df = pd.read_csv(csv_path)
        self.features = features
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        X = row[self.features].values.astype("float32")
        y = int(row[self.target_col])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--features-json", required=True, help='JSON with {"features": [...], "target": "label"}')
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, nargs="*", default=[32, 16])
    ap.add_argument("--out-dir", default="experiments")
    args = ap.parse_args()

    seed_everything(42)
    meta = json.load(open(args.features_json, "r", encoding="utf-8"))
    features = meta["features"]
    target = meta["target"]

    train_ds = TabularDataset(args.train_csv, features, target)
    val_ds = TabularDataset(args.val_csv, features, target)

    # fit scaler on train
    X_train = train_ds.df[features].values.astype("float32")
    scaler = fit_scaler(X_train)

    def collate(batch):
        X = torch.stack([b[0] for b in batch])
        y = torch.stack([b[1] for b in batch])
        X = (X - torch.tensor(scaler.mean_, dtype=torch.float32)) / torch.tensor(np.sqrt(scaler.var_), dtype=torch.float32)
        return X, y

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    num_classes = len(pd.unique(train_ds.df[target]))
    model = CognitiveMLP(len(features), num_classes, hidden_dims=args.hidden)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    run_id = f"cognitive_{timestamp()}"
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)

        # val
        model.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                loss_sum += loss.item() * X.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        val_loss = loss_sum / max(total, 1)
        val_acc = correct / max(total, 1)

        metrics = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        save_json(metrics, str(run_dir / f"epoch_{epoch:03d}.json"))

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(), "features": features, "target": target}, str(run_dir / "best_model.pth"))
            save_scaler(scaler, str(run_dir / "scaler.joblib"))

    save_json({"best_val_acc": best_acc}, str(run_dir / "results.json"))
    print(f"Training complete. Best val acc={best_acc:.4f}.")


if __name__ == "__main__":
    main()
