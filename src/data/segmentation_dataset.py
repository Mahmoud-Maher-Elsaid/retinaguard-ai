from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

LESION_FOLDERS = {
    "microaneurysms": "1. Microaneurysms",
    "haemorrhages": "2. Haemorrhages",
    "hard_exudates": "3. Hard Exudates",
    "soft_exudates": "4. Soft Exudates",
}


class IDRiDSegmentationDataset(Dataset):
    """
    IDRiD lesion segmentation dataset for U-Net.

    Output:
        image: [3, H, W]
        mask:  [4, H, W]

    Mask channels:
        0: Microaneurysms
        1: Haemorrhages
        2: Hard Exudates
        3: Soft Exudates
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        if split not in ["train", "test"]:
            raise ValueError("split must be either 'train' or 'test'")

        if split == "train":
            self.image_dir = self.root_dir / "1. Original Images" / "a. Training Set"
            self.mask_root = self.root_dir / "2. All Segmentation Groundtruths" / "a. Training Set"
        else:
            self.image_dir = self.root_dir / "1. Original Images" / "b. Testing Set"
            self.mask_root = self.root_dir / "2. All Segmentation Groundtruths" / "b. Testing Set"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if not self.mask_root.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_root}")

        self.image_paths = sorted([
            p for p in self.image_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in: {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _find_mask_path(self, image_stem: str, lesion_folder: str):
        folder = self.mask_root / lesion_folder

        if not folder.exists():
            return None

        candidates = [
            p for p in folder.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS and image_stem in p.stem
        ]

        if len(candidates) == 0:
            return None

        return sorted(candidates)[0]

    def _read_binary_mask(self, mask_path, target_shape):
        h, w = target_shape

        if mask_path is None:
            return np.zeros((h, w), dtype=np.float32)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            return np.zeros((h, w), dtype=np.float32)

        mask = (mask > 0).astype(np.float32)
        return mask

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_stem = image_path.stem

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        masks = []

        for _, folder_name in LESION_FOLDERS.items():
            mask_path = self._find_mask_path(image_stem, folder_name)
            mask = self._read_binary_mask(mask_path, target_shape=(h, w))
            masks.append(mask)

        mask = np.stack(masks, axis=-1).astype(np.float32)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)

        mask = mask.float()

        return {
            "image": image,
            "mask": mask,
            "image_id": image_stem,
            "split": self.split,
        }
