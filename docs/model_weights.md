# Model Weights and Checkpoints

Trained model weights, checkpoints, raw datasets, and experiment runs are not uploaded to GitHub.

Expected local checkpoint folders:
- checkpoints/classifier/
- checkpoints/segmentation/
- checkpoints/fusion/

The API configuration should point to local checkpoint paths through configs/api/example_api_config.yaml.

Ignored folders:
- data/raw/
- data/processed/
- checkpoints/
- weights/
- runs/
- wandb/
- mlruns/

For sharing large model files, use GitHub Releases, Google Drive, OneDrive, Hugging Face Hub, or Git LFS.
