# RetinaGuard-AI Dataset Card

Datasets used:
1. APTOS 2019 Blindness Detection
2. IDRiD

APTOS 2019:
- Train: 2930 images
- Validation: 366 images
- Test: 366 images

IDRiD:
- Grading train: 413 images
- Grading test: 103 images
- Segmentation train: 54 images
- Segmentation test: 27 images

DR classes:
0 - No DR
1 - Mild
2 - Moderate
3 - Severe
4 - Proliferative DR

Segmentation lesions:
- Microaneurysms
- Haemorrhages
- Hard Exudates
- Soft Exudates

Ignored data paths:
- data/raw/
- data/processed/
- checkpoints/

Dataset risks:
- Class imbalance
- Dataset shift
- Different camera/image quality conditions
- Small segmentation training set
