import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset_registry import (
    load_aptos_split,
    load_idrid_grading_split,
)
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
    parser.add_argument("--thresholds-csv", type=str, required=True)
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="unet_resnet34_tuned_thresholds")

    return parser.parse_args()


def load_thresholds(path: Path) -> np.ndarray:
    df = pd.read_csv(path)

    thresholds = np.zeros(4, dtype=np.float32)

    for _, row in df.iterrows():
        channel = int(row["channel"])
        thresholds[channel] = float(row["threshold"])

    return thresholds


def safe_path(path_value):
    path = Path(str(path_value))

    if path.exists():
        return path

    candidate = ROOT / path
    if candidate.exists():
        return candidate

    return path


def build_input_dataframe():
    frames = []

    for split in ["train", "valid", "test"]:
        df = load_aptos_split(split).copy()
        df["dataset"] = "APTOS2019"
        df["split"] = split
        frames.append(df)

    for split in ["train", "test"]:
        df = load_idrid_grading_split(split).copy()
        df["dataset"] = "IDRiD"
        df["split"] = split
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if "image_exists" in combined.columns:
        combined = combined[combined["image_exists"] == True].copy()

    combined["image_path"] = combined["image_path"].astype(str)
    combined = combined.reset_index(drop=True)

    return combined


class FundusImageDataset(Dataset):
    def __init__(self, dataframe, image_size):
        self.df = dataframe.reset_index(drop=True)
        self.transform = get_segmentation_transforms(image_size=image_size, train=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index].to_dict()

        image_path = safe_path(row["image_path"])
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image)
        image_tensor = transformed["image"]

        label = row.get("label", -1)
        if pd.isna(label):
            label = -1

        label_name = row.get("label_name", "unknown")
        if pd.isna(label_name):
            label_name = "unknown"

        return {
            "image": image_tensor,
            "image_id": str(row.get("image_id", image_path.stem)),
            "dataset": str(row.get("dataset", "unknown")),
            "split": str(row.get("split", "unknown")),
            "label": int(label),
            "label_name": str(label_name),
            "image_path": str(image_path),
        }


def connected_component_stats(binary_mask):
    binary = binary_mask.astype(np.uint8)

    if binary.sum() == 0:
        return 0, 0, 0.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    component_count = max(0, num_labels - 1)

    if component_count == 0:
        return 0, 0, 0.0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(areas.max())
    total_area = binary.shape[0] * binary.shape[1]
    largest_area_ratio = largest_area / total_area

    return component_count, largest_area, largest_area_ratio


def extract_stats_for_sample(prob_map, pred_mask):
    stats = {}

    h, w = pred_mask.shape[1], pred_mask.shape[2]
    total_pixels = h * w

    combined_mask = pred_mask.max(axis=0)
    union_area_pixels = int(combined_mask.sum())
    sum_area_pixels = int(pred_mask.sum())

    stats["total_lesion_union_area_pixels"] = union_area_pixels
    stats["total_lesion_union_area_ratio"] = union_area_pixels / total_pixels
    stats["total_lesion_sum_area_pixels"] = sum_area_pixels
    stats["total_lesion_sum_area_ratio"] = sum_area_pixels / total_pixels

    presence_count = 0

    for channel, lesion in enumerate(LESION_NAMES):
        binary = pred_mask[channel].astype(np.uint8)
        probs = prob_map[channel]

        area_pixels = int(binary.sum())
        area_ratio = area_pixels / total_pixels
        presence = int(area_pixels > 0)

        component_count, largest_area, largest_area_ratio = connected_component_stats(binary)

        presence_count += presence

        stats[f"{lesion}_area_pixels"] = area_pixels
        stats[f"{lesion}_area_ratio"] = area_ratio
        stats[f"{lesion}_presence"] = presence
        stats[f"{lesion}_prob_mean"] = float(probs.mean())
        stats[f"{lesion}_prob_max"] = float(probs.max())
        stats[f"{lesion}_component_count"] = component_count
        stats[f"{lesion}_largest_component_area_pixels"] = largest_area
        stats[f"{lesion}_largest_component_area_ratio"] = largest_area_ratio

    stats["lesion_presence_count"] = presence_count

    return stats


@torch.no_grad()
def main():
    args = parse_args()

    cv2.setNumThreads(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = ROOT / "reports" / "tables" / "lesion_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("RetinaGuard-AI Stage 6.1: Extract Lesion Statistics")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Thresholds CSV: {args.thresholds_csv}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"AMP: {args.amp}")
    print("=" * 100)

    thresholds = load_thresholds(Path(args.thresholds_csv))
    print("Thresholds:")
    for lesion, threshold in zip(LESION_NAMES, thresholds):
        print(f"  {lesion}: {threshold:.2f}")

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

    dataframe = build_input_dataframe()

    if args.max_images is not None:
        dataframe = dataframe.head(args.max_images).copy()

    print()
    print("Input images:")
    print(dataframe.groupby(["dataset", "split"]).size().to_string())
    print(f"Total images: {len(dataframe)}")
    print("=" * 100)

    dataset = FundusImageDataset(
        dataframe=dataframe,
        image_size=args.image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    threshold_tensor = torch.tensor(
        thresholds,
        dtype=torch.float32,
        device=device,
    ).view(1, 4, 1, 1)

    rows = []

    for batch in tqdm(loader, desc="Extract lesion features"):
        images = batch["image"].to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=args.amp and device.type == "cuda",
        ):
            logits = model(images)
            probs = torch.sigmoid(logits)

        pred_masks = (probs > threshold_tensor).float()

        probs_np = probs.detach().cpu().numpy()
        masks_np = pred_masks.detach().cpu().numpy()

        batch_size = images.size(0)

        for i in range(batch_size):
            base = {
                "dataset": batch["dataset"][i],
                "split": batch["split"][i],
                "image_id": batch["image_id"][i],
                "label": int(batch["label"][i]),
                "label_name": batch["label_name"][i],
                "image_path": batch["image_path"][i],
            }

            stats = extract_stats_for_sample(
                prob_map=probs_np[i],
                pred_mask=masks_np[i],
            )

            base.update(stats)
            rows.append(base)

    features_df = pd.DataFrame(rows)

    all_path = output_dir / f"{args.run_name}_lesion_features_all.csv"
    features_df.to_csv(all_path, index=False, encoding="utf-8-sig")

    for (dataset_name, split_name), group in features_df.groupby(["dataset", "split"]):
        safe_dataset = str(dataset_name).lower().replace(" ", "_")
        safe_split = str(split_name).lower().replace(" ", "_")

        split_path = output_dir / f"{args.run_name}_{safe_dataset}_{safe_split}_lesion_features.csv"
        group.to_csv(split_path, index=False, encoding="utf-8-sig")

    summary_rows = []

    feature_cols = [
        col for col in features_df.columns
        if col.endswith("_area_ratio") or col.endswith("_presence")
    ]

    for (dataset_name, split_name), group in features_df.groupby(["dataset", "split"]):
        row = {
            "dataset": dataset_name,
            "split": split_name,
            "num_images": len(group),
        }

        for col in feature_cols:
            row[f"mean_{col}"] = group[col].mean()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"{args.run_name}_lesion_feature_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print()
    print("Saved:")
    print(all_path)
    print(summary_path)
    print()
    print("Summary:")
    print(summary_df.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
