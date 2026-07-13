import argparse
import csv
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.segmentation_dataset import IDRiDSegmentationDataset
from src.data.transforms import get_segmentation_transforms
from src.models.segmentation_models import create_unet
from src.training.losses import BCEDiceLoss
from src.evaluation.segmentation_metrics import segmentation_metrics_from_logits


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--run-name", type=str, default="unet_resnet34_idrid")

    return parser.parse_args()


def seed_everything(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_train_val_indices(n, val_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    val_size = max(1, int(n * val_fraction))
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    return train_indices, val_indices


def save_history_row(csv_path, row):
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def aggregate_metric_dicts(metric_dicts):
    keys = metric_dicts[0].keys()
    return {
        key: float(np.mean([m[key] for m in metric_dicts]))
        for key in keys
    }


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()

    total_loss = 0.0
    metric_rows = []

    progress = tqdm(loader, desc="Train", leave=False)

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

        metric_rows.append(
            segmentation_metrics_from_logits(
                logits.detach(),
                masks.detach(),
                threshold=0.5,
            )
        )

        progress.set_postfix(loss=float(loss.item()))

    metrics = aggregate_metric_dicts(metric_rows)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, use_amp):
    model.eval()

    total_loss = 0.0
    metric_rows = []

    progress = tqdm(loader, desc="Valid", leave=False)

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True).float()

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
        else:
            logits = model(images)
            loss = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)

        metric_rows.append(
            segmentation_metrics_from_logits(
                logits.detach(),
                masks.detach(),
                threshold=0.5,
            )
        )

    metrics = aggregate_metric_dicts(metric_rows)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = ROOT / "data" / "raw" / "IDRiD" / "segmentation" / "A. Segmentation"

    run_dir = ROOT / "reports" / "tables" / args.run_name
    checkpoint_dir = ROOT / "checkpoints" / args.run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_path = run_dir / "history.csv"
    best_model_path = checkpoint_dir / "best_model.pt"
    last_model_path = checkpoint_dir / "last_model.pt"

    print("=" * 80)
    print("RetinaGuard-AI Stage 5: U-Net IDRiD Lesion Segmentation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder_name}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    print(f"AMP: {args.amp}")
    print(f"Run name: {args.run_name}")
    print("=" * 80)

    train_full = IDRiDSegmentationDataset(
        root_dir=root_dir,
        split="train",
        transform=get_segmentation_transforms(
            image_size=args.image_size,
            train=True,
        ),
    )

    valid_full = IDRiDSegmentationDataset(
        root_dir=root_dir,
        split="train",
        transform=get_segmentation_transforms(
            image_size=args.image_size,
            train=False,
        ),
    )

    train_indices, val_indices = make_train_val_indices(
        n=len(train_full),
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    train_dataset = Subset(train_full, train_indices)
    valid_dataset = Subset(valid_full, val_indices)

    print(f"Full IDRiD segmentation train images: {len(train_full)}")
    print(f"Train subset: {len(train_dataset)}")
    print(f"Validation subset: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    model = create_unet(
        encoder_name=args.encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=4,
    )

    model = model.to(device)

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

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

    best_dice = -1.0
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
            "train_mean_dice": train_metrics["mean_dice"],
            "train_mean_iou": train_metrics["mean_iou"],
            "valid_loss": valid_metrics["loss"],
            "valid_mean_dice": valid_metrics["mean_dice"],
            "valid_mean_iou": valid_metrics["mean_iou"],
            "valid_dice_microaneurysms": valid_metrics["dice_microaneurysms"],
            "valid_dice_haemorrhages": valid_metrics["dice_haemorrhages"],
            "valid_dice_hard_exudates": valid_metrics["dice_hard_exudates"],
            "valid_dice_soft_exudates": valid_metrics["dice_soft_exudates"],
            "valid_iou_microaneurysms": valid_metrics["iou_microaneurysms"],
            "valid_iou_haemorrhages": valid_metrics["iou_haemorrhages"],
            "valid_iou_hard_exudates": valid_metrics["iou_hard_exudates"],
            "valid_iou_soft_exudates": valid_metrics["iou_soft_exudates"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        save_history_row(history_path, row)

        print(
            f"Train loss={row['train_loss']:.4f} "
            f"dice={row['train_mean_dice']:.4f} "
            f"iou={row['train_mean_iou']:.4f}"
        )

        print(
            f"Valid loss={row['valid_loss']:.4f} "
            f"dice={row['valid_mean_dice']:.4f} "
            f"iou={row['valid_mean_iou']:.4f}"
        )

        print(
            "Valid Dice per lesion: "
            f"MA={row['valid_dice_microaneurysms']:.4f}, "
            f"HE={row['valid_dice_haemorrhages']:.4f}, "
            f"EX={row['valid_dice_hard_exudates']:.4f}, "
            f"SE={row['valid_dice_soft_exudates']:.4f}"
        )

        current_dice = valid_metrics["mean_dice"]

        checkpoint = {
            "epoch": epoch,
            "encoder_name": args.encoder_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "valid_metrics": valid_metrics,
            "args": vars(args),
        }

        torch.save(checkpoint, last_model_path)

        if current_dice > best_dice:
            best_dice = current_dice
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(checkpoint, best_model_path)

            print(f"New best U-Net saved. Best Dice={best_dice:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No Dice improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= args.patience:
            print()
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    elapsed_minutes = (time.time() - start_time) / 60

    print()
    print("=" * 80)
    print("U-Net training completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best valid Dice: {best_dice:.4f}")
    print(f"Elapsed minutes: {elapsed_minutes:.2f}")
    print(f"History: {history_path}")
    print(f"Best model: {best_model_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
