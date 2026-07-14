# RetinaGuard-AI Research Paper Draft

Title:
RetinaGuard-AI: An Explainable and Safety-Aware Pipeline for Diabetic Retinopathy Grading Using Lesion Segmentation and Late Fusion

Abstract:
RetinaGuard-AI is an end-to-end research pipeline combining EfficientNet-B0 diabetic retinopathy grading, U-Net lesion segmentation, lesion-derived feature analysis, late-fusion experiments, explainability visualizations, and an uncertainty-aware Safety Gate.

Methods:
The pipeline includes EfficientNet-B0 classification, U-Net segmentation, threshold tuning, lesion feature extraction, feature QA, lesion-only classification, late fusion, explainability, Safety Gate, and API deployment scaffold.

Main results:
- EfficientNet-B0 APTOS test QWK = 0.8899
- Probability-only meta-classifier APTOS test QWK = 0.8965
- U-Net tuned-threshold IDRiD test mean Dice = 0.4831
- Total lesion area Spearman correlation with DR grade = 0.7661

Limitations:
- Small IDRiD segmentation training set
- Weak microaneurysm segmentation
- Dataset shift
- No clinical validation

Conclusion:
RetinaGuard-AI goes beyond classification by adding lesion segmentation, lesion-derived evidence, fusion experiments, explainability, and conservative safety-aware triage.
