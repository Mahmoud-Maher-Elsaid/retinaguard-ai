import argparse
from pathlib import Path
import sys

import cv2
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
from src.evaluation.segmentation_metrics import segmentation_metrics_from_logits


LESION_NAMES = [
    "Microaneurysms",
    "Haemorrhages",
    "Hard Exudates",
    "Soft Exudates",
]

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

COLORS = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.2, 1.0],
    [1.0, 1.0, 0.0],
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--run-name", type=str, default="unet_resnet34_idrid_100ep")
    parser.add_argument("--max-visuals", type=int, default=8)
    return parser.parse_args()


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


def save_visual(image, true_mask, pred_mask, image_id, output_path):
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
    axes[1, 1].set_title("Predicted Combined Mask")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(pred_overlay)
    axes[1, 2].set_title("Predicted Overlay")
    axes[1, 2].axis("off")

    fig.suptitle(
        "U-Net Prediction Visual Check\n"
        "Red=Microaneurysms, Green=Haemorrhages, Blue=Hard Exudates, Yellow=Soft Exudates",
        fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_table_dir = ROOT / "reports" / "tables" / args.run_name
    output_fig_dir = ROOT / "reports" / "figures" / args.run_name / f"{args.split}_predictions"

    output_table_dir.mkdir(parents=True, exist_ok=True)
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RetinaGuard-AI U-Net Evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Threshold: {args.threshold}")

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

    rows = []
    visual_count = 0

    for batch in tqdm(loader, desc=f"Evaluate U-Net {args.split}"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True).float()

        logits = model(images)

        metrics = segmentation_metrics_from_logits(
            logits=logits,
            targets=masks,
            threshold=args.threshold,
        )

        probs = torch.sigmoid(logits)
        preds = (probs > args.threshold).float()

        for i in range(images.size(0)):
            image_id = batch["image_id"][i]

            row = {
                "image_id": image_id,
                **metrics,
            }

            rows.append(row)

            if visual_count < args.max_visuals:
                image = denormalize_image(batch["image"][i])
                true_mask = batch["mask"][i].detach().cpu().numpy()
                pred_mask = preds[i].detach().cpu().numpy()

                save_visual(
                    image=image,
                    true_mask=true_mask,
                    pred_mask=pred_mask,
                    image_id=image_id,
                    output_path=output_fig_dir / f"{image_id}_prediction_overlay.png",
                )

                visual_count += 1

    df = pd.DataFrame(rows)

    summary = {
        "split": args.split,
        "threshold": args.threshold,
        "mean_dice": df["mean_dice"].mean(),
        "mean_iou": df["mean_iou"].mean(),
        "dice_microaneurysms": df["dice_microaneurysms"].mean(),
        "dice_haemorrhages": df["dice_haemorrhages"].mean(),
        "dice_hard_exudates": df["dice_hard_exudates"].mean(),
        "dice_soft_exudates": df["dice_soft_exudates"].mean(),
        "iou_microaneurysms": df["iou_microaneurysms"].mean(),
        "iou_haemorrhages": df["iou_haemorrhages"].mean(),
        "iou_hard_exudates": df["iou_hard_exudates"].mean(),
        "iou_soft_exudates": df["iou_soft_exudates"].mean(),
        "checkpoint": args.checkpoint,
    }

    predictions_path = output_table_dir / f"{args.split}_segmentation_predictions.csv"
    metrics_path = output_table_dir / f"{args.split}_segmentation_metrics.csv"

    df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(metrics_path, index=False, encoding="utf-8-sig")

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
    print(metrics_path)
    print(output_fig_dir)

    print("=" * 80)


if __name__ == "__main__":
    main()
