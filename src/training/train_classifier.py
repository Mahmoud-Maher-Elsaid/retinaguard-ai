import argparse
import csv
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset_registry import load_aptos_split
from src.data.classification_dataset import RetinopathyClassificationDataset
from src.data.transforms import get_classification_transforms
from src.models.classification_models import create_classifier


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, default="efficientnet_b0")
    parser.add_argument("--dataset", type=str, default="aptos", choices=["aptos"])
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing value for CrossEntropyLoss.")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--run-name", type=str, default="efficientnet_b0_aptos")

    return parser.parse_args()


def seed_everything(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels, num_classes=5):
    labels = np.array(labels)
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = labels.shape[0] / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def create_weighted_sampler(labels, num_classes=5):
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()

    total_loss = 0.0
    all_targets = []
    all_preds = []

    progress = tqdm(loader, desc="Train", leave=False)

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits.detach(), dim=1)

        total_loss += loss.item() * images.size(0)
        all_targets.extend(targets.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

        progress.set_postfix(loss=float(loss.item()))

    avg_loss = total_loss / len(loader.dataset)

    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = avg_loss

    return metrics


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, use_amp):
    model.eval()

    total_loss = 0.0
    all_targets = []
    all_preds = []

    progress = tqdm(loader, desc="Valid", leave=False)

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
        else:
            logits = model(images)
            loss = criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * images.size(0)
        all_targets.extend(targets.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)

    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = avg_loss

    return metrics


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


def save_history_row(csv_path, row):
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    args = parse_args()
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = ROOT / "reports" / "tables" / args.run_name
    checkpoint_dir = ROOT / "checkpoints" / args.run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_path = run_dir / "history.csv"
    best_model_path = checkpoint_dir / "best_model.pt"
    last_model_path = checkpoint_dir / "last_model.pt"

    print("=" * 80)
    print("RetinaGuard-AI Stage 4: EfficientNet DR Grading Baseline")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    print(f"AMP: {args.amp}")
    print(f"Weighted sampler: {args.weighted_sampler}")
    print(f"Run name: {args.run_name}")
    print("=" * 80)

    train_df = load_aptos_split("train")
    valid_df = load_aptos_split("valid")

    train_transform = get_classification_transforms(
        image_size=args.image_size,
        train=True,
    )

    valid_transform = get_classification_transforms(
        image_size=args.image_size,
        train=False,
    )

    train_dataset = RetinopathyClassificationDataset(
        dataframe=train_df,
        transform=train_transform,
    )

    valid_dataset = RetinopathyClassificationDataset(
        dataframe=valid_df,
        transform=valid_transform,
    )

    train_labels = train_df["label"].astype(int).tolist()
    class_weights = compute_class_weights(train_labels, num_classes=5).to(device)

    if args.weighted_sampler:
        sampler = create_weighted_sampler(train_labels, num_classes=5)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_classifier(
        model_name=args.model_name,
        num_classes=5,
        pretrained=not args.no_pretrained,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    best_qwk = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print()
        print(f"Epoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=args.amp,
        )

        valid_metrics = validate_one_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            use_amp=args.amp,
        )

        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_qwk": train_metrics["qwk"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_weighted_f1": valid_metrics["weighted_f1"],
            "valid_qwk": valid_metrics["qwk"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        save_history_row(history_path, row)

        print(
            f"Train loss={row['train_loss']:.4f} "
            f"acc={row['train_accuracy']:.4f} "
            f"macro_f1={row['train_macro_f1']:.4f} "
            f"qwk={row['train_qwk']:.4f}"
        )

        print(
            f"Valid loss={row['valid_loss']:.4f} "
            f"acc={row['valid_accuracy']:.4f} "
            f"macro_f1={row['valid_macro_f1']:.4f} "
            f"qwk={row['valid_qwk']:.4f}"
        )

        current_qwk = valid_metrics["qwk"]

        checkpoint = {
            "epoch": epoch,
            "model_name": args.model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "valid_metrics": valid_metrics,
            "args": vars(args),
        }

        torch.save(checkpoint, last_model_path)

        if current_qwk > best_qwk:
            best_qwk = current_qwk
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(checkpoint, best_model_path)

            print(f"New best model saved. Best QWK={best_qwk:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No QWK improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= args.patience:
            print()
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    elapsed_minutes = (time.time() - start_time) / 60

    print()
    print("=" * 80)
    print("Training completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best valid QWK: {best_qwk:.4f}")
    print(f"Elapsed minutes: {elapsed_minutes:.2f}")
    print(f"History: {history_path}")
    print(f"Best model: {best_model_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
