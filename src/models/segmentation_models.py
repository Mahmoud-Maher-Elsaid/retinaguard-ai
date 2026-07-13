import segmentation_models_pytorch as smp


def create_unet(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    num_classes: int = 4,
):
    """
    Create U-Net segmentation model for IDRiD lesion segmentation.

    Output channels:
        0: Microaneurysms
        1: Haemorrhages
        2: Hard Exudates
        3: Soft Exudates
    """

    try:
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
    except Exception as e:
        print(f"WARNING: Could not create U-Net with encoder_weights={encoder_weights}. Reason: {e}")
        print("Falling back to encoder_weights=None.")
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

    return model


def create_unetplusplus(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    num_classes: int = 4,
):
    """
    Optional stronger segmentation baseline: U-Net++.
    """

    try:
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
    except Exception as e:
        print(f"WARNING: Could not create U-Net++ with encoder_weights={encoder_weights}. Reason: {e}")
        print("Falling back to encoder_weights=None.")
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

    return model


def create_deeplabv3plus(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    num_classes: int = 4,
):
    """
    Optional segmentation baseline: DeepLabV3+.
    """

    try:
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
    except Exception as e:
        print(f"WARNING: Could not create DeepLabV3+ with encoder_weights={encoder_weights}. Reason: {e}")
        print("Falling back to encoder_weights=None.")
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

    return model
