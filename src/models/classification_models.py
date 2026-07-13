import timm
import torch.nn as nn


def create_classifier(
    model_name: str = "efficientnet_b0",
    num_classes: int = 5,
    pretrained: bool = True,
):
    """
    Create a DR grading classifier using timm.

    If pretrained weights fail to download, fallback to random initialization.
    """

    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except Exception as e:
        print(f"WARNING: Could not load pretrained={pretrained}. Reason: {e}")
        print("Falling back to pretrained=False.")
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
        )

    return model
