# Dataset Split Metadata

This folder is reserved for small dataset split metadata files used to reproduce the RetinaGuard-AI experiments.

Raw fundus images, segmentation masks, generated overlays, trained checkpoints, and large experiment artifacts must not be stored here.

## Purpose

Dataset split files make experiments reproducible by recording exactly which images belong to each training, validation, and test subset.

This is important because model performance can change significantly if dataset splits are regenerated differently.

## Allowed Files

Only small metadata files should be committed in this folder.

Recommended files:

- aptos_train.csv
- aptos_val.csv
- aptos_test.csv
- idrid_grading_train.csv
- idrid_grading_test.csv
- idrid_segmentation_train.csv
- idrid_segmentation_test.csv
- idrid_localization_train.csv
- idrid_localization_test.csv

## Forbidden Files

Do not commit:

- raw images
- segmentation masks
- model checkpoints
- generated predictions
- experiment runs
- zip files
- cached tensors
- large CSV exports with duplicated image data

## Recommended CSV Schema

For classification or grading splits:

| column | description |
|---|---|
| image_id | Unique image identifier without leaking patient-level information |
| image_path | Relative local path from the dataset root |
| label | Numeric class label |
| label_name | Human-readable class name |
| split | train, val, or test |
| dataset | Dataset name, such as APTOS2019 or IDRiD |

For segmentation splits:

| column | description |
|---|---|
| image_id | Unique image identifier |
| image_path | Relative path to the fundus image |
| mask_path | Relative path to the lesion mask or mask folder |
| lesion_type | microaneurysms, haemorrhages, hard_exudates, or soft_exudates |
| split | train, val, or test |
| dataset | Dataset name |

## Example Classification Split

```csv
image_id,image_path,label,label_name,split,dataset
000c1434d8d7,train_images/000c1434d8d7.png,2,Moderate,train,APTOS2019
001639a390f0,train_images/001639a390f0.png,4,Proliferative DR,val,APTOS2019
0024cdab0c1e,train_images/0024cdab0c1e.png,1,Mild,test,APTOS2019
```

## Example Segmentation Split

```csv
image_id,image_path,mask_path,lesion_type,split,dataset
IDRiD_01,Original Images/Training Set/IDRiD_01.jpg,Groundtruths/Training Set/Microaneurysms/IDRiD_01_MA.tif,microaneurysms,train,IDRiD
IDRiD_02,Original Images/Training Set/IDRiD_02.jpg,Groundtruths/Training Set/Haemorrhages/IDRiD_02_HE.tif,haemorrhages,train,IDRiD
IDRiD_03,Original Images/Testing Set/IDRiD_03.jpg,Groundtruths/Testing Set/Hard Exudates/IDRiD_03_EX.tif,hard_exudates,test,IDRiD
```

## Reproducibility Rules

- Keep split files small and text-based.
- Use relative paths, not absolute machine-specific paths.
- Do not include personal, private, or patient-identifying information.
- Do not regenerate splits silently after reporting results.
- If a split is changed, document why it changed in the commit message or experiment notes.
- Keep class labels consistent with the training and evaluation scripts.
- Keep the same split files when comparing multiple models.
- Do not mix validation data with test data during threshold tuning or model selection.

## Validation Checklist

Before committing split metadata, verify:

- The file size is small.
- No raw images, masks, checkpoints, or zip files are staged.
- All labels match the project label mapping.
- All referenced files exist locally.
- All paths are relative to the expected dataset root.
- The split column contains only train, val, or test.
- The dataset column clearly identifies the source dataset.
- The split files do not contain private or patient-identifying information.

## Current Status

This repository currently documents the expected split metadata structure only.

The actual raw datasets are stored locally and are intentionally excluded from GitHub.

Related documentation:

- docs/data_setup.md
- docs/model_weights.md
- docs/reproduce_results.md