# RetinaGuard-AI Final Experiment Summary

## Main Classification Result
- Best EfficientNet-B0 baseline: `no_sampler` with APTOS test QWK `0.8899`, accuracy `0.8361`, macro F1 `0.6495`, weighted F1 `0.8321`.

## Main Segmentation Result
- Tuned-threshold U-Net ResNet34 achieved IDRiD test mean Dice `0.4831` and mean IoU `0.3480`.
- Per-lesion Dice: MA `0.0712`, Haemorrhages `0.5717`, Hard Exudates `0.6896`, Soft Exudates `0.5999`.

## Lesion Feature Findings
- U-Net-derived lesion features showed strong correlation with DR grade.
- Lesion-feature-only classifiers were weaker than image-based EfficientNet, but confirmed that lesion statistics contain useful disease signal.

## Late Fusion Finding
- Best APTOS test late-fusion/meta-classifier result: `prob_only_logistic_balanced` with QWK `0.8965`, accuracy `0.8388`, macro F1 `0.6531`.
- Lesion fusion produced mixed gains, so the final claim should be conservative and honest.

## Final Project Claim
RetinaGuard-AI provides an end-to-end diabetic retinopathy pipeline combining image-based grading, lesion segmentation, lesion-derived statistical analysis, and late-fusion experiments. The strongest grading performance comes from EfficientNet-based image classification, while U-Net lesion features provide interpretable medical evidence and useful auxiliary signals.