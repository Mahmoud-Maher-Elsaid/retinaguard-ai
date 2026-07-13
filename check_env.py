import sys
import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import timm
import albumentations as A
import segmentation_models_pytorch as smp
import torchmetrics
import monai

print("=" * 70)
print("RetinaGuard-AI Environment Check")
print("=" * 70)

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch CUDA:", torch.version.cuda)

    x = torch.randn(2, 3, 224, 224).cuda()
    model = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).cuda()
    y = model(x)

    print("GPU tensor test:", y.shape)
    print("Allocated GPU memory MB:", round(torch.cuda.memory_allocated() / 1024 / 1024, 2))
else:
    print("WARNING: CUDA is not available. Training will run on CPU.")

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("MONAI:", monai.__version__)

print("=" * 70)
print("Testing EfficientNet-B0 from timm...")
clf = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
print("EfficientNet-B0 created successfully.")

print("=" * 70)
print("Testing ResNet34 U-Net...")
unet = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=4
)
print("ResNet34-U-Net created successfully.")

if torch.cuda.is_available():
    clf = clf.cuda()
    unet = unet.cuda()
    test_img_cls = torch.randn(1, 3, 224, 224).cuda()
    test_img_seg = torch.randn(1, 3, 512, 512).cuda()

    with torch.no_grad():
        cls_out = clf(test_img_cls)
        seg_out = unet(test_img_seg)

    print("Classification output shape:", cls_out.shape)
    print("Segmentation output shape:", seg_out.shape)

print("=" * 70)
print("Environment is ready for RetinaGuard-AI.")
print("=" * 70)
