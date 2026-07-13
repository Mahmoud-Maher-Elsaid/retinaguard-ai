import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.segmentation_dataset import IDRiDSegmentationDataset
from src.data.transforms import get_segmentation_transforms
from src.models.segmentation_models import create_unet


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

LESION_NAMES = [
    "microaneurysms",
    "haemorrhages",
    "hard_exudates",
    "soft_exudates",
]

COLORS = np.array([
    [1.0, 0.0, 0.0],    # Microaneurysms - Red
    [0.0, 1.0, 0.0],    # Haemorrhages - Green
    [0.0, 0.2, 1.0],    # Hard Exudates - Blue
    [1.0, 1.0, 0.0],    # Soft Exudates - Yellow
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--thresholds-csv", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--run-name", type=str, default="unet_resnet34_idrid_100ep")
    parser.add_argument("--max-visuals", type=int, default=12)
    return parser.parse_args()


def load_thresholds(path: Path) -> np.ndarray:
    df = pd.read_csv(path)

    thresholds = np.zeros(4, dtype=np.float32)

    for _, row in df.iterrows():
        channel = int(row["channel"])
        thresholds[channel] = float(row["threshold"])

    return thresholds


def denormalize_image(image_tensor):
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * STD + MEAN
    image = np.clip(image, 0, 1)
    return image


def make_overlay(image, mask, alpha=0.45):
    overlay = image.copy()

    for c in range(mask.shape[0]):
        binary = mask[c] > 0.5
        color = COLORS[c]
        overlay[binary] = (1 - alpha) * overlay[binary] + alpha * color

    return np.clip(overlay, 0, 1)


def dice_iou_per_channel(pred_mask, true_mask, smooth=1.0):
    rows = []

    for c, lesion in enumerate(LESION_NAMES):
        pred = pred_mask[c].float()
        true = true_mask[c].float()

        intersection = torch.sum(pred * true)
        pred_sum = torch.sum(pred)
        true_sum = torch.sum(true)

        dice = (2.0 * intersection + smooth) / (pred_sum + true_sum + smooth)

        union = pred_sum + true_sum - intersection
        iou = (intersection + smooth) / (union + smooth)

        rows.append({
            "lesion": lesion,
            "channel": c,
            "dice": float(dice.detach().cpu()),
            "iou": float(iou.detach().cpu()),
        })

    return rows


def save_visual(image, true_mask, pred_mask, image_id, thresholds, output_path):
    true_overlay = make_overlay(image, true_mask)
    pred_overlay = make_overlay(image, pred_mask)

    true_combined = true_mask.max(axis=0)
    pred_combined = pred_mask.max(axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Image: {image_id}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(true_combined, cmap="gray")
    axes[0, 1].set_title("Ground Truth Combined Mask")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(true_overlay)
    axes[0, 2].set_title("Ground Truth Overlay")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(image)
    axes[1, 0].set_title("Image")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_combined, cmap="gray")
    axes[1, 1].set_title("Predicted Combined Mask - Tuned Thresholds")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(pred_overlay)
    axes[1, 2].set_title("Predicted Overlay - Tuned Thresholds")
    axes[1, 2].axis("off")

    threshold_text = (
        f"MA={thresholds[0]:.2f}, HE={thresholds[1]:.2f}, "
        f"EX={thresholds[2]:.2f}, SE={thresholds[3]:.2f}"
    )

    fig.suptitle(
        "U-Net Tuned Threshold Prediction Check\n"
        "Red=Microaneurysms, Green=Haemorrhages, Blue=Hard Exudates, Yellow=Soft Exudates\n"
        f"Thresholds: {threshold_text}",
        fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    thresholds = load_thresholds(Path(args.thresholds_csv))

    output_table_dir = ROOT / "reports" / "tables" / args.run_name
    output_fig_dir = ROOT / "reports" / "figures" / args.run_name / f"{args.split}_predictions_tuned_thresholds"

    output_table_dir.mkdir(parents=True, exist_ok=True)
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RetinaGuard-AI U-Net Tuned Threshold Visual Evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Thresholds: {thresholds.tolist()}")

    root_dir = ROOT / "data" / "raw" / "IDRiD" / "segmentation" / "A. Segmentation"

    dataset = IDRiDSegmentationDataset(
        root_dir=root_dir,
        split=args.split,
        transform=get_segmentation_transforms(image_size=args.image_size, train=False),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

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

    metric_rows = []
    visual_count = 0

    threshold_tensor = torch.tensor(thresholds, dtype=torch.float32).view(1, 4, 1, 1).to(device)

    for batch in tqdm(loader, desc=f"Evaluate tuned U-Net {args.split}"):
        images = batch["image"].to(device, non_blocking=True)
        true_masks = batch["mask"].to(device, non_blocking=True).float()

        logits = model(images)
        probs = torch.sigmoid(logits)
        pred_masks = (probs > threshold_tensor).float()

        for i in range(images.size(0)):
            image_id = batch["image_id"][i]

            rows = dice_iou_per_channel(
                pred_mask=pred_masks[i].detach().cpu(),
                true_mask=true_masks[i].detach().cpu(),
            )

            for row in rows:
                row["image_id"] = image_id
                row["split"] = args.split
                row["threshold"] = thresholds[row["channel"]]
                metric_rows.append(row)

            if visual_count < args.max_visuals:
                image = denormalize_image(batch["image"][i])
                true_mask_np = batch["mask"][i].detach().cpu().numpy()
                pred_mask_np = pred_masks[i].detach().cpu().numpy()

                save_visual(
                    image=image,
                    true_mask=true_mask_np,
                    pred_mask=pred_mask_np,
                    image_id=image_id,
                    thresholds=thresholds,
                    output_path=output_fig_dir / f"{image_id}_tuned_prediction_overlay.png",
                )

                visual_count += 1

    metrics_df = pd.DataFrame(metric_rows)

    per_lesion = (
        metrics_df.groupby(["lesion", "channel", "threshold"])[["dice", "iou"]]
        .mean()
        .reset_index()
    )

    summary = {
        "split": args.split,
        "mean_dice": per_lesion["dice"].mean(),
        "mean_iou": per_lesion["iou"].mean(),
    }

    for _, row in per_lesion.iterrows():
        lesion = row["lesion"]
        summary[f"dice_{lesion}"] = row["dice"]
        summary[f"iou_{lesion}"] = row["iou"]
        summary[f"threshold_{lesion}"] = row["threshold"]

    predictions_path = output_table_dir / f"{args.split}_tuned_threshold_visual_metrics_per_image.csv"
    per_lesion_path = output_table_dir / f"{args.split}_tuned_threshold_visual_metrics_per_lesion.csv"
    summary_path = output_table_dir / f"{args.split}_tuned_threshold_visual_summary.csv"

    metrics_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    per_lesion.to_csv(per_lesion_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print()
    print("Per-lesion metrics:")
    print(per_lesion.to_string(index=False))

    print()
    print("Summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print()
    print("Saved:")
    print(predictions_path)
    print(per_lesion_path)
    print(summary_path)
    print(output_fig_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
