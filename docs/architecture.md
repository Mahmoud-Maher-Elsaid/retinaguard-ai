# RetinaGuard-AI Architecture

RetinaGuard-AI is designed as a multi-stage diabetic retinopathy analysis framework.

```
mermaid
flowchart TD
    A[Input Fundus Image] --> B[Image Preprocessing]
    B --> C[EfficientNet-B0 DR Classifier]
    B --> D[U-Net ResNet34 Lesion Segmentation]
    C --> E[Class Probabilities and DR Grade]
    D --> F[Lesion Masks]
    F --> G[Lesion Feature Extraction]
    E --> H[Late Fusion Model]
    G --> H
    H --> I[Final DR Prediction]
    E --> J[Uncertainty and Confidence Check]
    G --> K[Lesion Evidence Check]
    I --> L[Safety-Aware Triage Gate]
    J --> L
    K --> L
    D --> M[Mask-to-Box Lesion Localization]
    L --> N[FastAPI Backend]
    M --> N
    N --> O[Web Interface Output]
```

## Main Components

- EfficientNet-B0 classifier for diabetic retinopathy grading.
- U-Net ResNet34 model for lesion segmentation.
- Lesion feature extraction from segmentation masks.
- Late fusion between classifier probabilities and lesion evidence.
- Safety-aware triage gate for manual review routing.
- FastAPI backend and web interface for demonstration.

## Safety Philosophy

The system is conservative by design. It is not a certified medical device and should not be used for real diagnosis. Uncertain, low-quality, or inconsistent cases should be routed to manual review.
