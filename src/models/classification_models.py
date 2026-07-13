import timm


def create_classifier(
    model_name: str = "efficientnet_b0",
    num_classes: int = 5,
    pretrained: bool = True,
):
    """
    Create a diabetic retinopathy grading classifier using timm.

    Examples:
        efficientnet_b0
        convnext_tiny
        resnet50
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
