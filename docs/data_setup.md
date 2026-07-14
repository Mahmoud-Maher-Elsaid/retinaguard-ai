# RetinaGuard-AI Data Setup

Raw medical image datasets are not uploaded to GitHub.

The datasets should be stored locally under:

D:\retinaguard-ai\data\raw

Local ZIP source:

C:\Users\maher\Downloads

Required ZIP files:

- archive (2).zip
- A. Segmentation.zip
- B. Disease Grading.zip
- C. Localization.zip

Expected local layout:

data/raw/APTOS2019/
data/raw/IDRiD/segmentation/
data/raw/IDRiD/disease_grading/
data/raw/IDRiD/localization/

GitHub intentionally ignores:

- data/raw/
- data/processed/
- archives/
- checkpoints/
- weights/
- runs/
- .venv/

This keeps the repository clean and reproducible without uploading raw datasets or training artifacts.
