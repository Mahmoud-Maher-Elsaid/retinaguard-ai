# RetinaGuard-AI Deployment Guide

This guide explains how to run the RetinaGuard-AI single-image inference pipeline and the optional FastAPI research service.

Important notice: RetinaGuard-AI is a research prototype. It is not clinically validated and must not be used for diagnosis or medical decision-making.

Single-image inference:
python src\inference\retinaguard_single_image.py --image path\to\fundus_image.png --amp --output-json reports\sample_prediction.json

Install API dependencies:
pip install -r requirements-api.txt

Start the API:
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Open the API docs:
http://127.0.0.1:8000/docs

Expected local checkpoints:
checkpoints/efficientnet_b0_aptos_100ep_bs16_no_sampler/best_model.pt
checkpoints/unet_resnet34_idrid_100ep/best_model.pt

Checkpoints are intentionally not committed to GitHub.
