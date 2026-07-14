# RetinaGuard-AI Model Card

Model name: RetinaGuard-AI

Model type:
- EfficientNet-B0 diabetic retinopathy classifier
- U-Net ResNet34 lesion segmentation model
- Lesion feature extraction
- Late fusion experiments
- Explainability demo
- Safety Gate triage layer

Intended use:
Research and educational use in diabetic retinopathy analysis.

Not intended for:
- Clinical diagnosis
- Treatment decisions
- Autonomous medical screening
- Replacing clinicians

Main results:
- EfficientNet-B0 APTOS test QWK: 0.8899
- Probability-only meta-classifier APTOS test QWK: 0.8965
- U-Net tuned-threshold IDRiD test mean Dice: 0.4831
- Lesion area vs DR grade Spearman correlation: 0.7661

Known limitations:
- Microaneurysm segmentation is weak.
- IDRiD segmentation set is small.
- Dataset shift exists between APTOS and IDRiD.
- The project is not clinically validated.

Safety note:
RetinaGuard-AI is a research prototype only and must not be used for clinical decision-making.
