from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.segmentation_dataset import IDRiDSegmentationDataset
from src.data.transforms import get_segmentation_transforms


def inspect_segmentation_loader(split="train", image_size=512, batch_size=2):
    print("=" * 80)
    print(f"Testing IDRiD Segmentation DataLoader: {split}")
    print("=" * 80)

    root_dir = ROOT / "data" / "raw" / "IDRiD" / "segmentation" / "A. Segmentation"

    dataset = IDRiDSegmentationDataset(
        root_dir=root_dir,
        split=split,
        transform=get_segmentation_transforms(image_size=image_size, train=(split == "train")),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(loader))

    print("Dataset size:", len(dataset))
    print("Batch image shape:", batch["image"].shape)
    print("Batch mask shape:", batch["mask"].shape)
    print("Batch image IDs:", list(batch["image_id"]))

    mask = batch["mask"]
    print("Mask dtype:", mask.dtype)
    print("Mask min:", float(mask.min()))
    print("Mask max:", float(mask.max()))
    print("Positive pixels per channel:", mask.sum(dim=(0, 2, 3)).tolist())

    if torch.cuda.is_available():
        device = torch.device("cuda")
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        print("Moved batch to GPU:", images.device)
        print("GPU image tensor shape:", images.shape)
        print("GPU mask tensor shape:", masks.shape)

    print("Segmentation DataLoader test passed.")


def main():
    print("=" * 80)
    print("RetinaGuard-AI Stage 3.2: Segmentation DataLoader Test")
    print("=" * 80)

    inspect_segmentation_loader(split="train", image_size=512, batch_size=2)
    inspect_segmentation_loader(split="test", image_size=512, batch_size=2)

    print("=" * 80)
    print("Stage 3.2 segmentation DataLoader completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
