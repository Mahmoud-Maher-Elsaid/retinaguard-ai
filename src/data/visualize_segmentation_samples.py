from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.segmentation_dataset import IDRiDSegmentationDataset
from src.data.transforms import get_segmentation_transforms


REPORT_FIGURES = ROOT / "reports" / "figures"
REPORT_FIGURES.mkdir(parents=True, exist_ok=True)

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


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * STD + MEAN
    image = np.clip(image, 0, 1)
    return image


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image.copy()

    for channel_idx in range(mask.shape[0]):
        binary_mask = mask[channel_idx] > 0.5
        color = COLORS[channel_idx]

        overlay[binary_mask] = (
            (1 - alpha) * overlay[binary_mask] + alpha * color
        )

    return np.clip(overlay, 0, 1)


def plot_samples(split: str, output_path: Path, max_samples: int = 4) -> None:
    root_dir = ROOT / "data" / "raw" / "IDRiD" / "segmentation" / "A. Segmentation"

    dataset = IDRiDSegmentationDataset(
        root_dir=root_dir,
        split=split,
        transform=get_segmentation_transforms(image_size=512, train=False),
    )

    n = min(max_samples, len(dataset))

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(13, 4 * n))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx in range(n):
        sample = dataset[idx]

        image = denormalize_image(sample["image"])
        mask = sample["mask"].detach().cpu().numpy()

        if mask.shape[0] != 4:
            raise ValueError(f"Expected mask shape [4, H, W], got {mask.shape}")

        combined_mask = mask.max(axis=0)
        overlay = create_overlay(image, mask)

        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f"{split} image: {sample['image_id']}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(combined_mask, cmap="gray")
        axes[idx, 1].set_title("Combined lesion mask")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title("Overlay")
        axes[idx, 2].axis("off")

    fig.suptitle(
        "IDRiD Segmentation Visual Sanity Check\n"
        "Red=Microaneurysms, Green=Haemorrhages, Blue=Hard Exudates, Yellow=Soft Exudates",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    print("=" * 80)
    print("RetinaGuard-AI Stage 3.3: Segmentation Visual Sanity Check")
    print("=" * 80)

    plot_samples(
        split="train",
        output_path=REPORT_FIGURES / "idrid_segmentation_visual_check_train.png",
        max_samples=4,
    )

    plot_samples(
        split="test",
        output_path=REPORT_FIGURES / "idrid_segmentation_visual_check_test.png",
        max_samples=4,
    )

    print("=" * 80)
    print("Visual sanity check completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
