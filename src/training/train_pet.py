import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.vit_pet import create_vit_model, get_transforms
from src.training.datasets import ImageCSVDataset
from src.training.common import seed_everything, train_one_epoch, validate, save_json, timestamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--classes-json", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--model-name", default="vit_base_patch16_224")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--out-dir", default="experiments")
    ap.add_argument("--limit-train", type=int, default=0, help="Limit number of training samples (0=no limit)")
    ap.add_argument("--limit-val", type=int, default=0, help="Limit number of validation samples (0=no limit)")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--use-weighted-sampler", action="store_true")
    ap.add_argument("--use-class-weights", action="store_true")
    ap.add_argument("--mixup-alpha", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    seed_everything(42)
    with open(args.classes_json, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    train_ds = ImageCSVDataset(args.train_csv, args.classes_json, transform=get_transforms(args.image_size, True))
    val_ds = ImageCSVDataset(args.val_csv, args.classes_json, transform=get_transforms(args.image_size, False))

    from torch.utils.data import Subset
    if args.limit_train and args.limit_train > 0:
        train_ds = Subset(train_ds, list(range(min(len(train_ds), args.limit_train))))
    if args.limit_val and args.limit_val > 0:
        val_ds = Subset(val_ds, list(range(min(len(val_ds), args.limit_val))))

    from torch.utils.data import Subset
    from src.training.common import make_weighted_sampler_for_dataset
    if args.limit_train and args.limit_train > 0:
        train_ds = Subset(train_ds, list(range(min(len(train_ds), args.limit_train))))
    if args.limit_val and args.limit_val > 0:
        val_ds = Subset(val_ds, list(range(min(len(val_ds), args.limit_val))))

    if args.use_weighted_sampler:
        sampler = make_weighted_sampler_for_dataset(train_ds, args.classes_json)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_vit_model(num_classes=num_classes, model_name=args.model_name, pretrained=args.pretrained)
    model.to(device)

    if args.use_class_weights:
        from src.training.common import compute_class_weights
        cw = compute_class_weights(args.train_csv, args.classes_json).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    run_id = f"pet_{timestamp()}"
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_path = run_dir / "best_model.pth"

    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, mixup_alpha=args.mixup_alpha)
        val_loss, report, cm = validate(model, val_loader, criterion, device, [idx_to_class[i] for i in range(num_classes)])
        macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
        scheduler.step()
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_macro_f1": macro_f1,
            "classification_report": report,
            "confusion_matrix": cm,
        }
        save_json(metrics, str(run_dir / f"epoch_{epoch:03d}.json"))
        if macro_f1 >= best_f1:
            best_f1 = macro_f1
            torch.save({"model_state": model.state_dict(), "classes": class_to_idx}, str(best_path))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break
    save_json({"best_macro_f1": best_f1, "best_model": str(best_path)}, str(run_dir / "results.json"))
    print(f"Training complete. Best macro F1={best_f1:.4f}. Best model at {best_path}")


if __name__ == "__main__":
    main()
