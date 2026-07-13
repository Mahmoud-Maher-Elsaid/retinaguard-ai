from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class RetinopathyClassificationDataset(Dataset):
    """
    PyTorch Dataset for diabetic retinopathy grading.

    Expected DataFrame columns:
        - dataset
        - split
        - image_id
        - label
        - label_name
        - image_path
        - image_exists
    """

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transform = transform

        required_columns = [
            "dataset",
            "split",
            "image_id",
            "label",
            "label_name",
            "image_path",
            "image_exists",
        ]

        missing_columns = [col for col in required_columns if col not in self.df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        missing_images = self.df[~self.df["image_exists"]]

        if len(missing_images) > 0:
            raise FileNotFoundError(
                f"{len(missing_images)} images are missing. "
                "Run src/data/verify_datasets.py first."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = Path(row["image_path"])

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "image": image,
            "label": label,
            "image_id": row["image_id"],
            "dataset": row["dataset"],
            "split": row["split"],
            "label_name": row["label_name"],
        }
