# RetinaGuard-AI

**RetinaGuard-AI** is an end-to-end diabetic retinopathy analysis project that combines image-based grading, lesion segmentation, lesion-derived feature analysis, and late-fusion experiments.

The project is designed as a research-style, GitHub-ready medical AI pipeline.

> **Important:** This project is for research and educational purposes only. It is not a medical diagnosis system and must not be used for clinical decisions.

---

## Project Motivation

Most diabetic retinopathy projects only classify the disease grade from a fundus image.

RetinaGuard-AI goes further by asking:

> Can lesion segmentation provide interpretable evidence and auxiliary features for diabetic retinopathy grading?

The project studies:

1. **Image-level grading** using EfficientNet-B0.
2. **Lesion segmentation** using U-Net ResNet34.
3. **Lesion-derived statistics** for interpretability.
4. **Late fusion** between EfficientNet probabilities and U-Net lesion features.

---

## Main Pipeline

```text
Fundus Image
    |
    |-- EfficientNet-B0
    |       -> DR grade prediction
    |       -> class probabilities
    |
    |-- U-Net ResNet34
    |       -> Microaneurysm mask
    |       -> Haemorrhage mask
    |       -> Hard Exudate mask
    |       -> Soft Exudate mask
    |
    |-- Lesion Feature Extraction
    |       -> lesion area ratios
    |       -> component counts
    |       -> probability statistics
    |
    |-- Lesion Feature QA
    |       -> correlation with DR grade
    |
    |-- Late Fusion
            -> EfficientNet probabilities + lesion features
            -> meta-classifier
```

---

## Datasets

### APTOS 2019 Blindness Detection

Used mainly for diabetic retinopathy grading.

```text
APTOS train: 2930 images
APTOS valid: 366 images
APTOS test:  366 images
```

### IDRiD

Used for lesion segmentation and external grading checks.

```text
IDRiD grading train: 413 images
IDRiD grading test:  103 images

IDRiD segmentation train: 54 images
IDRiD segmentation test:  27 images
```

Raw datasets and checkpoints are intentionally ignored by Git:

```text
data/raw/
data/processed/
checkpoints/
```

---

## Diabetic Retinopathy Classes

```text
0 - No DR
1 - Mild
2 - Moderate
3 - Severe
4 - Proliferative DR
```

---

## Final Results

## 1. EfficientNet-B0 Classification

| Experiment | Accuracy | Macro F1 | Weighted F1 | QWK |
| --- | --- | --- | --- | --- |
| weighted_sampler | 0.8251 | 0.6500 | 0.8241 | 0.8880 |
| no_sampler | 0.8361 | 0.6495 | 0.8321 | 0.8899 |
| no_sampler_lowlr_smooth | 0.8306 | 0.6513 | 0.8271 | 0.8708 |

### Best image-only baseline

```text
Model: no_sampler
APTOS test QWK: 0.8899
APTOS test accuracy: 0.8361
```

---

## 2. U-Net Lesion Segmentation

| Model | Mean Dice | Mean IoU | MA Dice | HE Dice | EX Dice | SE Dice |
| --- | --- | --- | --- | --- | --- | --- |
| U-Net ResNet34 raw threshold 0.5 | 0.3595 | 0.2686 | 0.0079 | 0.4950 | 0.5664 | 0.3689 |
| U-Net ResNet34 validation-tuned thresholds | 0.4831 | 0.3480 | 0.0712 | 0.5717 | 0.6896 | 0.5999 |

### Best segmentation result

```text
Model: U-Net ResNet34 validation-tuned thresholds
IDRiD test mean Dice: 0.4831
Hard Exudates Dice: 0.6896
Soft Exudates Dice: 0.5999
```

Microaneurysm segmentation remains the weakest lesion channel.

---

## 3. Lesion Feature Quality Analysis

U-Net-derived lesion features showed strong correlation with diabetic retinopathy grade.

| Feature | Spearman Corr. | Pearson Corr. |
| --- | --- | --- |
| total_lesion_union_area_ratio | 0.7661 | 0.5691 |
| hard_exudates_area_ratio | 0.7198 | 0.3960 |
| microaneurysms_area_ratio | 0.7065 | 0.5809 |
| haemorrhages_area_ratio | 0.6800 | 0.4291 |
| soft_exudates_area_ratio | 0.3546 | 0.3419 |
| lesion_presence_count | 0.3405 | 0.3250 |

This supports the idea that predicted lesion statistics are medically meaningful and useful for interpretability.

---

## 4. Lesion-Feature-Only Classifier

A classifier was trained using only U-Net-derived lesion features.

| Model | Accuracy | Macro F1 | Weighted F1 | QWK |
| --- | --- | --- | --- | --- |
| logistic_regression_balanced | 0.6967 | 0.5406 | 0.7167 | 0.7752 |
| random_forest_balanced | 0.7514 | 0.4978 | 0.7330 | 0.7623 |
| extra_trees_balanced | 0.7596 | 0.5332 | 0.7451 | 0.7468 |

The lesion-feature-only classifier is weaker than EfficientNet, but it confirms that lesion statistics contain useful disease signal.

---

## 5. Late Fusion Experiments

Late fusion combines EfficientNet probabilities with U-Net lesion features.

| Model | Feature Set | Accuracy | Macro F1 | Weighted F1 | QWK |
| --- | --- | --- | --- | --- | --- |
| efficientnet_raw | image_only | 0.8361 | 0.6495 | 0.8321 | 0.8899 |
| prob_only_logistic_balanced | prob_only | 0.8388 | 0.6531 | 0.8350 | 0.8965 |
| late_fusion_logistic_balanced | fusion | 0.8443 | 0.6673 | 0.8390 | 0.8869 |
| late_fusion_random_forest_balanced | fusion | 0.8415 | 0.6539 | 0.8345 | 0.8915 |
| late_fusion_extra_trees_balanced | fusion | 0.8470 | 0.6716 | 0.8412 | 0.8883 |

### Best meta-classifier result

```text
Model: prob_only_logistic_balanced
Feature set: prob_only
APTOS test QWK: 0.8965
```

The probability-only meta-classifier achieved the highest QWK. Lesion fusion improved some metrics such as accuracy and macro F1, but did not consistently dominate QWK.


---

## 6. RetinaGuard Safety Gate

RetinaGuard-AI includes an uncertainty-aware safety triage layer.

This layer combines EfficientNet confidence, prediction entropy, top-2 probability margin, fundus image quality, U-Net lesion burden, and lesion-grade consistency.

It produces a safety-aware decision instead of only a class prediction:

- safe_negative_prediction
- low_risk_follow_up
- follow_up_recommended
- routine_referral
- urgent_referral
- manual_review_required

### Safety Gate Summary

| Triage Decision | Count |
|---|---:|
| Manual review required | 2773 |
| Safe negative prediction | 655 |
| Urgent referral | 191 |
| Routine referral | 77 |
| Follow-up recommended | 65 |
| Low-risk follow-up | 4 |

This stage makes RetinaGuard-AI more realistic because it asks whether a prediction is safe to trust or should be reviewed manually.

---

## Key Findings

| Section | Finding | Decision |
| --- | --- | --- |
| Classification | Best EfficientNet-B0 overall model is no-sampler. | Use efficientnet_b0_aptos_100ep_bs16_no_sampler as the main image-only baseline. |
| Classification | Label smoothing improved macro F1 slightly but reduced QWK. | Keep label smoothing as an ablation, not the final baseline. |
| Segmentation | Validation-tuned thresholds improved U-Net segmentation substantially. | Use tuned-threshold segmentation results in README and qualitative figures. |
| Segmentation | Microaneurysm segmentation remains the weakest lesion channel. | Report this honestly as a limitation and future improvement target. |
| Lesion features | Lesion features alone contain useful disease signal but are weaker than EfficientNet. | Use lesion features as interpretable auxiliary features, not as a replacement for image models. |
| Late fusion | Probability-only meta-classifier achieved the highest APTOS test QWK. | Treat this as calibration/meta-classification, not lesion-based improvement. |
| Late fusion | Lesion fusion improved some metrics but did not consistently dominate QWK. | Report fusion results as mixed: useful for interpretability and some metric gains, but image-only EfficientNet remains very strong. |
| Lesion feature QA | Total predicted lesion area strongly correlates with DR grade. | Use this as the main evidence that U-Net-derived lesion features are medically meaningful. |

---

## Project Structure

```text
retinaguard-ai/
│
├── configs/
├── notebooks/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── utils/
│
├── reports/
│   ├── figures/
│   ├── tables/
│   └── final_experiment_summary.md
│
├── data/
│   ├── raw/          # ignored
│   └── splits/
│
├── checkpoints/     # ignored
├── README.md
└── .gitignore
```

---

## Reproducing the Experiments

### 1. Environment check

```bash
python check_env.py
```

### 2. Verify datasets

```bash
python src/data/verify_datasets.py
```

### 3. Run EDA

```bash
python src/data/run_eda.py
```

### 4. Train EfficientNet-B0 classifier

```bash
python src/training/train_classifier.py \
    --model-name efficientnet_b0 \
    --image-size 384 \
    --batch-size 16 \
    --epochs 100 \
    --patience 12 \
    --num-workers 2 \
    --amp \
    --run-name efficientnet_b0_aptos_100ep_bs16_no_sampler
```

### 5. Evaluate classifier

```bash
python src/evaluation/evaluate_classifier.py \
    --checkpoint checkpoints/efficientnet_b0_aptos_100ep_bs16_no_sampler/best_model.pt \
    --split test \
    --image-size 384 \
    --batch-size 16 \
    --run-name efficientnet_b0_aptos_100ep_bs16_no_sampler
```

### 6. Train U-Net

```bash
python src/training/train_unet.py \
    --encoder-name resnet34 \
    --image-size 512 \
    --batch-size 2 \
    --epochs 100 \
    --patience 15 \
    --amp \
    --run-name unet_resnet34_idrid_100ep
```

### 7. Tune U-Net thresholds

```bash
python src/evaluation/tune_unet_thresholds.py \
    --checkpoint checkpoints/unet_resnet34_idrid_100ep/best_model.pt \
    --encoder-name resnet34 \
    --image-size 512 \
    --batch-size 1 \
    --run-name unet_resnet34_idrid_100ep
```

### 8. Extract lesion features

```bash
python src/evaluation/extract_lesion_features.py \
    --checkpoint checkpoints/unet_resnet34_idrid_100ep/best_model.pt \
    --thresholds-csv reports/tables/unet_resnet34_idrid_100ep/best_thresholds_validation.csv \
    --encoder-name resnet34 \
    --image-size 512 \
    --batch-size 2 \
    --amp \
    --run-name unet_resnet34_tuned_thresholds
```

### 9. Analyze lesion features

```bash
python src/evaluation/analyze_lesion_features.py \
    --features-csv reports/tables/lesion_features/unet_resnet34_tuned_thresholds_lesion_features_all.csv \
    --output-name unet_resnet34_tuned_thresholds
```

### 10. Train lesion-feature-only classifier

```bash
python src/evaluation/train_lesion_feature_classifier.py \
    --features-csv reports/tables/lesion_features/unet_resnet34_tuned_thresholds_lesion_features_all.csv \
    --run-name lesion_feature_classifier_aptos_train
```

### 11. Train late fusion classifier

```bash
python src/evaluation/train_late_fusion_classifier.py \
    --efficientnet-checkpoint checkpoints/efficientnet_b0_aptos_100ep_bs16_no_sampler/best_model.pt \
    --lesion-features-csv reports/tables/lesion_features/unet_resnet34_tuned_thresholds_lesion_features_all.csv \
    --model-name efficientnet_b0 \
    --image-size 384 \
    --batch-size 16 \
    --amp \
    --run-name late_fusion_effnet_b0_lesion_features
```

### 12. Build final summary

```bash
python src/evaluation/build_final_summary.py
```

---

## Reports

Important result files:

```text
reports/final_experiment_summary.md

reports/tables/final_summary/final_classification_results.csv
reports/tables/final_summary/final_segmentation_results.csv
reports/tables/final_summary/final_lesion_feature_classifier_results.csv
reports/tables/final_summary/final_late_fusion_results.csv
reports/tables/final_summary/final_lesion_feature_correlations.csv
reports/tables/final_summary/final_key_findings.csv
```

Important visualizations:

```text
reports/figures/unet_resnet34_idrid_100ep/test_predictions_tuned_thresholds/
reports/figures/lesion_feature_qa/
reports/figures/late_fusion_effnet_b0_lesion_features/
```

---

## Limitations

1. Microaneurysm segmentation is weak.
2. IDRiD segmentation has only 54 training images.
3. Late fusion gives mixed gains and does not consistently dominate EfficientNet QWK.
4. External IDRiD grading performance is much lower than APTOS performance, suggesting dataset shift.
5. The project is not clinically validated.

---

## Future Work

- Use higher resolution segmentation such as 768x768 or patch-based training.
- Add Focal Tversky loss for small lesion segmentation.
- Improve microaneurysm detection with specialized small-object training.
- Add ConvNeXt or Swin Transformer classification baselines.
- Add stronger calibration methods.
- Train on combined APTOS + IDRiD grading data.
- Add Grad-CAM and lesion-guided explainability maps.
- Build a FastAPI inference backend.
- Build a simple clinical dashboard.

---

## Final Project Claim

RetinaGuard-AI provides an end-to-end diabetic retinopathy pipeline combining image-based grading, lesion segmentation, lesion-derived statistical analysis, and late-fusion experiments.

The strongest grading performance comes from EfficientNet-based image classification, while U-Net lesion features provide interpretable medical evidence and useful auxiliary signals.

---

## Author

Mahmoud Maher El-Said

Artificial Intelligence - Intelligent Systems  
Arab Academy for Science, Technology & Maritime Transport

