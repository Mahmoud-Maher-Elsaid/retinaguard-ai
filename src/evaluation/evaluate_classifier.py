import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset_registry import load_aptos_split, DR_CLASS_NAMES
from src.data.classification_dataset import RetinopathyClassificationDataset
from src.data.transforms import get_classification_transforms
from src.models.classification_models import create_classifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="efficientnet_b0_aptos_100ep_bs16_w2")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = ROOT / "reports" / "tables" / args.run_name
    fig_dir = ROOT / "reports" / "figures" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RetinaGuard-AI Classifier Evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model_name = checkpoint.get("model_name", ckpt_args.get("model_name", "efficientnet_b0"))

    model = create_classifier(
        model_name=model_name,
        num_classes=5,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    df = load_aptos_split(args.split)

    dataset = RetinopathyClassificationDataset(
        dataframe=df,
        transform=get_classification_transforms(image_size=args.image_size, train=False),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    rows = []
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc=f"Evaluate {args.split}"):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        for i in range(images.size(0)):
            true_label = int(labels[i].cpu())
            pred_label = int(preds[i].cpu())

            row = {
                "image_id": batch["image_id"][i],
                "true_label": true_label,
                "true_label_name": DR_CLASS_NAMES.get(true_label, "Unknown"),
                "pred_label": pred_label,
                "pred_label_name": DR_CLASS_NAMES.get(pred_label, "Unknown"),
                "correct": true_label == pred_label,
            }

            for c in range(5):
                row[f"prob_class_{c}"] = float(probs[i, c].cpu())

            rows.append(row)
            y_true.append(true_label)
            y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    print()
    print("Metrics:")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"QWK:         {qwk:.4f}")

    pred_df = pd.DataFrame(rows)
    pred_path = output_dir / f"{args.split}_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=[DR_CLASS_NAMES[i] for i in range(5)],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_path = output_dir / f"{args.split}_classification_report.csv"
    report_df.to_csv(report_path, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{i}_{DR_CLASS_NAMES[i]}" for i in range(5)],
        columns=[f"pred_{i}_{DR_CLASS_NAMES[i]}" for i in range(5)],
    )
    cm_csv_path = output_dir / f"{args.split}_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path, encoding="utf-8-sig")

    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title(f"APTOS {args.split} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(5), [f"{i}\n{DR_CLASS_NAMES[i]}" for i in range(5)], rotation=30, ha="right")
    plt.yticks(range(5), [f"{i}\n{DR_CLASS_NAMES[i]}" for i in range(5)])

    for i in range(5):
        for j in range(5):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    cm_fig_path = fig_dir / f"{args.split}_confusion_matrix.png"
    plt.savefig(cm_fig_path, dpi=200)
    plt.close()

    metrics_path = output_dir / f"{args.split}_metrics.csv"
    pd.DataFrame([{
        "split": args.split,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "qwk": qwk,
        "checkpoint": args.checkpoint,
    }]).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print()
    print("Saved:")
    print(pred_path)
    print(report_path)
    print(cm_csv_path)
    print(cm_fig_path)
    print(metrics_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
