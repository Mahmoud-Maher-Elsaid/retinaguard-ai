from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def read_csv(rel):
    return pd.read_csv(ROOT / rel)

def f4(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)

def table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)

classification = read_csv("reports/tables/final_summary/final_classification_results.csv")
segmentation = read_csv("reports/tables/final_summary/final_segmentation_results.csv")
fusion = read_csv("reports/tables/final_summary/final_late_fusion_results.csv")
lesion_clf = read_csv("reports/tables/final_summary/final_lesion_feature_classifier_results.csv")
corr = read_csv("reports/tables/final_summary/final_lesion_feature_correlations.csv")
findings = read_csv("reports/tables/final_summary/final_key_findings.csv")

cls_rows = []
for _, r in classification.iterrows():
    cls_rows.append([
        r["model"],
        f4(r["accuracy"]),
        f4(r["macro_f1"]),
        f4(r["weighted_f1"]),
        f4(r["qwk"]),
    ])

seg_rows = []
for _, r in segmentation.iterrows():
    seg_rows.append([
        r["model"],
        f4(r["mean_dice"]),
        f4(r["mean_iou"]),
        f4(r["dice_microaneurysms"]),
        f4(r["dice_haemorrhages"]),
        f4(r["dice_hard_exudates"]),
        f4(r["dice_soft_exudates"]),
    ])

fusion_test = fusion[fusion["split"] == "aptos_test"].copy()
fusion_rows = []
for _, r in fusion_test.iterrows():
    fusion_rows.append([
        r["model"],
        r["feature_set"],
        f4(r["accuracy"]),
        f4(r["macro_f1"]),
        f4(r["weighted_f1"]),
        f4(r["qwk"]),
    ])

lesion_test = lesion_clf[lesion_clf["split"] == "aptos_test"].copy()
lesion_rows = []
for _, r in lesion_test.iterrows():
    lesion_rows.append([
        r["model"],
        f4(r["accuracy"]),
        f4(r["macro_f1"]),
        f4(r["weighted_f1"]),
        f4(r["qwk"]),
    ])

corr_all = corr[corr["dataset"] == "ALL"].copy()
corr_rows = []
for _, r in corr_all.iterrows():
    corr_rows.append([
        r["feature"],
        f4(r["spearman_corr_with_grade"]),
        f4(r["pearson_corr_with_grade"]),
    ])

finding_rows = []
for _, r in findings.iterrows():
    finding_rows.append([
        r["section"],
        r["finding"],
        r["decision"],
    ])

best_cls = classification.sort_values("qwk", ascending=False).iloc[0]
best_fusion = fusion_test.sort_values("qwk", ascending=False).iloc[0]
best_seg = segmentation.sort_values("mean_dice", ascending=False).iloc[0]

readme = f"""# RetinaGuard-AI

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

{table(["Experiment", "Accuracy", "Macro F1", "Weighted F1", "QWK"], cls_rows)}

### Best image-only baseline

```text
Model: {best_cls["model"]}
APTOS test QWK: {f4(best_cls["qwk"])}
APTOS test accuracy: {f4(best_cls["accuracy"])}
```

---

## 2. U-Net Lesion Segmentation

{table(["Model", "Mean Dice", "Mean IoU", "MA Dice", "HE Dice", "EX Dice", "SE Dice"], seg_rows)}

### Best segmentation result

```text
Model: {best_seg["model"]}
IDRiD test mean Dice: {f4(best_seg["mean_dice"])}
Hard Exudates Dice: {f4(best_seg["dice_hard_exudates"])}
Soft Exudates Dice: {f4(best_seg["dice_soft_exudates"])}
```

Microaneurysm segmentation remains the weakest lesion channel.

---

## 3. Lesion Feature Quality Analysis

U-Net-derived lesion features showed strong correlation with diabetic retinopathy grade.

{table(["Feature", "Spearman Corr.", "Pearson Corr."], corr_rows)}

This supports the idea that predicted lesion statistics are medically meaningful and useful for interpretability.

---

## 4. Lesion-Feature-Only Classifier

A classifier was trained using only U-Net-derived lesion features.

{table(["Model", "Accuracy", "Macro F1", "Weighted F1", "QWK"], lesion_rows)}

The lesion-feature-only classifier is weaker than EfficientNet, but it confirms that lesion statistics contain useful disease signal.

---

## 5. Late Fusion Experiments

Late fusion combines EfficientNet probabilities with U-Net lesion features.

{table(["Model", "Feature Set", "Accuracy", "Macro F1", "Weighted F1", "QWK"], fusion_rows)}

### Best meta-classifier result

```text
Model: {best_fusion["model"]}
Feature set: {best_fusion["feature_set"]}
APTOS test QWK: {f4(best_fusion["qwk"])}
```

The probability-only meta-classifier achieved the highest QWK. Lesion fusion improved some metrics such as accuracy and macro F1, but did not consistently dominate QWK.

---

## Key Findings

{table(["Section", "Finding", "Decision"], finding_rows)}

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
python src/training/train_classifier.py \\
    --model-name efficientnet_b0 \\
    --image-size 384 \\
    --batch-size 16 \\
    --epochs 100 \\
    --patience 12 \\
    --num-workers 2 \\
    --amp \\
    --run-name efficientnet_b0_aptos_100ep_bs16_no_sampler
```

### 5. Evaluate classifier

```bash
python src/evaluation/evaluate_classifier.py \\
    --checkpoint checkpoints/efficientnet_b0_aptos_100ep_bs16_no_sampler/best_model.pt \\
    --split test \\
    --image-size 384 \\
    --batch-size 16 \\
    --run-name efficientnet_b0_aptos_100ep_bs16_no_sampler
```

### 6. Train U-Net

```bash
python src/training/train_unet.py \\
    --encoder-name resnet34 \\
    --image-size 512 \\
    --batch-size 2 \\
    --epochs 100 \\
    --patience 15 \\
    --amp \\
    --run-name unet_resnet34_idrid_100ep
```

### 7. Tune U-Net thresholds

```bash
python src/evaluation/tune_unet_thresholds.py \\
    --checkpoint checkpoints/unet_resnet34_idrid_100ep/best_model.pt \\
    --encoder-name resnet34 \\
    --image-size 512 \\
    --batch-size 1 \\
    --run-name unet_resnet34_idrid_100ep
```

### 8. Extract lesion features

```bash
python src/evaluation/extract_lesion_features.py \\
    --checkpoint checkpoints/unet_resnet34_idrid_100ep/best_model.pt \\
    --thresholds-csv reports/tables/unet_resnet34_idrid_100ep/best_thresholds_validation.csv \\
    --encoder-name resnet34 \\
    --image-size 512 \\
    --batch-size 2 \\
    --amp \\
    --run-name unet_resnet34_tuned_thresholds
```

### 9. Analyze lesion features

```bash
python src/evaluation/analyze_lesion_features.py \\
    --features-csv reports/tables/lesion_features/unet_resnet34_tuned_thresholds_lesion_features_all.csv \\
    --output-name unet_resnet34_tuned_thresholds
```

### 10. Train lesion-feature-only classifier

```bash
python src/evaluation/train_lesion_feature_classifier.py \\
    --features-csv reports/tables/lesion_features/unet_resnet34_tuned_thresholds_lesion_features_all.csv \\
    --run-name lesion_feature_classifier_aptos_train
```

### 11. Train late fusion classifier

```bash
python src/evaluation/train_late_fusion_classifier.py \\
    --efficientnet-checkpoint checkpoints/efficientnet_b0_aptos_100ep_bs16_no_sampler/best_model.pt \\
    --lesion-features-csv reports/tables/lesion_features/unet_resnet34_tuned_thresholds_lesion_features_all.csv \\
    --model-name efficientnet_b0 \\
    --image-size 384 \\
    --batch-size 16 \\
    --amp \\
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
"""

(ROOT / "README.md").write_text(readme, encoding="utf-8")
print("README.md updated successfully.")
