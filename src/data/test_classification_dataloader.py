from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset_registry import load_aptos_split, load_idrid_grading_split
from src.data.transforms import get_classification_transforms
from src.data.classification_dataset import RetinopathyClassificationDataset


def inspect_loader(name, dataframe, image_size=384, batch_size=4):
    print("=" * 80)
    print(f"Testing DataLoader: {name}")
    print("=" * 80)

    transform = get_classification_transforms(
        image_size=image_size,
        train=True,
    )

    dataset = RetinopathyClassificationDataset(
        dataframe=dataframe,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(loader))

    print("Dataset size:", len(dataset))
    print("Batch image shape:", batch["image"].shape)
    print("Batch label shape:", batch["label"].shape)
    print("Batch labels:", batch["label"].tolist())
    print("Batch image IDs:", list(batch["image_id"]))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        print("Moved batch to GPU:", images.device)
        print("GPU image tensor shape:", images.shape)
        print("GPU label tensor shape:", labels.shape)

    print("DataLoader test passed.")


def main():
    print("=" * 80)
    print("RetinaGuard-AI Stage 3: Classification DataLoader Test")
    print("=" * 80)

    aptos_train = load_aptos_split("train")
    idrid_train = load_idrid_grading_split("train")

    inspect_loader(
        name="APTOS2019 train classification",
        dataframe=aptos_train,
        image_size=384,
        batch_size=4,
    )

    inspect_loader(
        name="IDRiD train classification",
        dataframe=idrid_train,
        image_size=384,
        batch_size=4,
    )

    print("=" * 80)
    print("Stage 3 classification DataLoader completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
