# Model Weights and Checkpoints

RetinaGuard-AI does not upload trained model weights, checkpoints, or experiment runs directly to GitHub.

This is intentional because model checkpoints are usually large binary files and can make the repository heavy.

## Ignored weight folders

The following folders are ignored by Git:

- checkpoints/
- weights/
- runs/
- wandb/
- mlruns/

## Expected local usage

Place trained checkpoints locally under:

D:\retinaguard-ai\checkpoints

Recommended local structure:

- checkpoints/classifier/
- checkpoints/segmentation/
- checkpoints/fusion/

## Main model components

The system may use the following local artifacts depending on the experiment or API configuration:

- EfficientNet-B0 classifier checkpoint for diabetic retinopathy grading.
- U-Net ResNet34 segmentation checkpoint for lesion segmentation.
- Optional fusion model artifacts for combining classifier probabilities with lesion features.
- Optional safety or calibration artifacts if generated in later experiments.

## API configuration

The API should point to local model paths through configuration files under:

configs/api/

Example file:

configs/api/example_api_config.yaml

Before running the API on a new machine, update the config file so it points to the correct local checkpoint paths.

## Why weights are not committed

Raw datasets and trained weights are not committed because they are large and may have licensing or storage limitations.

For public sharing, use one of these options:

- GitHub Releases for small release artifacts.
- Google Drive or OneDrive for large checkpoints.
- Hugging Face Hub for model hosting.
- Git LFS only if you intentionally want versioned large files.

## Reproducibility

A user can reproduce the models by preparing the datasets locally, installing requirements, and running the training scripts in src/training/.

The repository stores source code, reports, documentation, the LaTeX paper, and the compiled paper PDF. It does not store raw datasets or full training checkpoints.
