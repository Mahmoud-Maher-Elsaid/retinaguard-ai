# Reproduce Results

This guide explains how to reproduce the main RetinaGuard-AI workflow on a local machine.

## 1. Create environment

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

## 2. Prepare datasets

Raw datasets are not uploaded to GitHub. Place them under:

D:\retinaguard-ai\data\raw

Expected dataset layout is documented in:

docs/data_setup.md

## 3. Verify datasets

python src\data\verify_datasets.py

## 4. Train classifier

python src\training\train_classifier.py

## 5. Evaluate classifier

python src\evaluation\evaluate_classifier.py

## 6. Train segmentation model

python src\training\train_unet.py

## 7. Evaluate segmentation model

python src\evaluation\evaluate_unet.py

## 8. Run safety triage report

python src\evaluation\generate_safety_triage_report.py

## 9. Run API demo

uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Then open:

http://localhost:8000

## Notes

Exact command arguments may vary depending on the selected experiment config and local checkpoint paths. Model weights are documented in docs/model_weights.md.
