import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_classification_transforms(image_size: int = 384, train: bool = True):
    """
    Albumentations preprocessing pipeline for fundus image classification.
    Output image tensor shape: [3, image_size, image_size]
    """

    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.3,
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ])

    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_segmentation_transforms(image_size: int = 512, train: bool = True):
    """
    Albumentations preprocessing pipeline for fundus lesion segmentation.
    Output:
        image tensor shape: [3, image_size, image_size]
        mask tensor shape:  [4, image_size, image_size]
    """

    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.12,
                contrast_limit=0.12,
                p=0.25,
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(transpose_mask=True),
        ])

    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(transpose_mask=True),
    ])
