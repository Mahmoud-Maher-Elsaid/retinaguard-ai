import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.segmentation_dataset import IDRiDSegmentationDataset
from src.data.transforms import get_segmentation_transforms
from src.models.segmentation_models import create_unet


LESION_NAMES = [
    "microaneurysms",
    "haemorrhages",
    "hard_exudates",
    "soft_exudates",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="unet_resnet34_idrid_100ep")

    return parser.parse_args()


def make_train_val_indices(n, val_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    val_size = max(1, int(n * val_fraction))
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    return train_indices, val_indices


@torch.no_grad()
def collect_probs_and_targets(model, loader, device):
    all_probs = []
    all_targets = []

    for batch in tqdm(loader, desc="Collect predictions"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True).float()

        logits = model(images)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu())
        all_targets.append(masks.detach().cpu())

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return probs, targets


def dice_for_channel(probs, targets, channel, threshold, smooth=1.0):
    pred = (probs[:, channel] > threshold).float()
    true = targets[:, channel].float()

    intersection = torch.sum(pred * true)
    pred_sum = torch.sum(pred)
    true_sum = torch.sum(true)

    dice = (2.0 * intersection + smooth) / (pred_sum + true_sum + smooth)

    return float(dice)


def iou_for_channel(probs, targets, channel, threshold, smooth=1.0):
    pred = (probs[:, channel] > threshold).float()
    true = targets[:, channel].float()

    intersection = torch.sum(pred * true)
    union = torch.sum(pred) + torch.sum(true) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return float(iou)


def tune_thresholds(probs, targets, thresholds):
    rows = []
    best_thresholds = {}

    for channel, lesion_name in enumerate(LESION_NAMES):
        best_dice = -1.0
        best_threshold = 0.5
        best_iou = 0.0

        for threshold in thresholds:
            dice = dice_for_channel(probs, targets, channel, threshold)
            iou = iou_for_channel(probs, targets, channel, threshold)

            rows.append({
                "lesion": lesion_name,
                "channel": channel,
                "threshold": threshold,
                "dice": dice,
                "iou": iou,
            })

            if dice > best_dice:
                best_dice = dice
                best_iou = iou
                best_threshold = threshold

        best_thresholds[lesion_name] = {
            "channel": channel,
            "threshold": best_threshold,
            "dice": best_dice,
            "iou": best_iou,
        }

    return pd.DataFrame(rows), best_thresholds


def evaluate_with_thresholds(probs, targets, best_thresholds):
    rows = []

    dice_values = []
    iou_values = []

    for channel, lesion_name in enumerate(LESION_NAMES):
        threshold = best_thresholds[lesion_name]["threshold"]

        dice = dice_for_channel(probs, targets, channel, threshold)
        iou = iou_for_channel(probs, targets, channel, threshold)

        dice_values.append(dice)
        iou_values.append(iou)

        rows.append({
            "lesion": lesion_name,
            "channel": channel,
            "threshold": threshold,
            "dice": dice,
            "iou": iou,
        })

    summary = {
        "mean_dice": float(np.mean(dice_values)),
        "mean_iou": float(np.mean(iou_values)),
        "dice_microaneurysms": dice_values[0],
        "dice_haemorrhages": dice_values[1],
        "dice_hard_exudates": dice_values[2],
        "dice_soft_exudates": dice_values[3],
        "iou_microaneurysms": iou_values[0],
        "iou_haemorrhages": iou_values[1],
        "iou_hard_exudates": iou_values[2],
        "iou_soft_exudates": iou_values[3],
    }

    return pd.DataFrame(rows), summary


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = ROOT / "reports" / "tables" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RetinaGuard-AI U-Net Threshold Tuning")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print("Tuning split: validation subset from IDRiD segmentation train")
    print("Final evaluation split: IDRiD segmentation test")
    print("=" * 80)

    root_dir = ROOT / "data" / "raw" / "IDRiD" / "segmentation" / "A. Segmentation"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder_name = checkpoint.get("encoder_name", args.encoder_name)

    model = create_unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        num_classes=4,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    valid_full = IDRiDSegmentationDataset(
        root_dir=root_dir,
        split="train",
        transform=get_segmentation_transforms(image_size=args.image_size, train=False),
    )

    _, val_indices = make_train_val_indices(
        n=len(valid_full),
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    valid_dataset = Subset(valid_full, val_indices)

    test_dataset = IDRiDSegmentationDataset(
        root_dir=root_dir,
        split="test",
        transform=get_segmentation_transforms(image_size=args.image_size, train=False),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Validation subset size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    valid_probs, valid_targets = collect_probs_and_targets(model, valid_loader, device)
    test_probs, test_targets = collect_probs_and_targets(model, test_loader, device)

    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2).tolist()

    grid_df, best_thresholds = tune_thresholds(
        probs=valid_probs,
        targets=valid_targets,
        thresholds=thresholds,
    )

    best_df = pd.DataFrame([
        {
            "lesion": lesion_name,
            **values,
        }
        for lesion_name, values in best_thresholds.items()
    ])

    val_eval_df, val_summary = evaluate_with_thresholds(
        probs=valid_probs,
        targets=valid_targets,
        best_thresholds=best_thresholds,
    )

    test_eval_df, test_summary = evaluate_with_thresholds(
        probs=test_probs,
        targets=test_targets,
        best_thresholds=best_thresholds,
    )

    grid_path = output_dir / "threshold_tuning_grid_validation.csv"
    best_path = output_dir / "best_thresholds_validation.csv"
    val_eval_path = output_dir / "validation_metrics_tuned_thresholds.csv"
    test_eval_path = output_dir / "test_metrics_tuned_thresholds_per_lesion.csv"
    test_summary_path = output_dir / "test_metrics_tuned_thresholds_summary.csv"

    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")
    val_eval_df.to_csv(val_eval_path, index=False, encoding="utf-8-sig")
    test_eval_df.to_csv(test_eval_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([test_summary]).to_csv(test_summary_path, index=False, encoding="utf-8-sig")

    print()
    print("Best thresholds selected on validation:")
    print(best_df.to_string(index=False))

    print()
    print("Validation summary with tuned thresholds:")
    for k, v in val_summary.items():
        print(f"{k}: {v:.4f}")

    print()
    print("Test summary with tuned thresholds:")
    for k, v in test_summary.items():
        print(f"{k}: {v:.4f}")

    print()
    print("Saved:")
    print(grid_path)
    print(best_path)
    print(val_eval_path)
    print(test_eval_path)
    print(test_summary_path)

    print("=" * 80)


if __name__ == "__main__":
    main()
