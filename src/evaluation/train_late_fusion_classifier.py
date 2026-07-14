import argparse
from pathlib import Path
import sys

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset_registry import load_aptos_split, load_idrid_grading_split
from src.data.transforms import get_classification_transforms
from src.models.classification_models import create_classifier


DR_LABELS = [0, 1, 2, 3, 4]
DR_LABEL_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

METADATA_COLUMNS = {
    "dataset",
    "split",
    "image_id",
    "label",
    "label_name",
    "image_path",
    "image_exists",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--efficientnet-checkpoint", type=str, required=True)
    parser.add_argument(
        "--lesion-features-csv",
        type=str,
        default="reports/tables/lesion_features/unet_resnet34_tuned_thresholds_lesion_features_all.csv",
    )
    parser.add_argument("--model-name", type=str, default="efficientnet_b0")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--run-name", type=str, default="late_fusion_effnet_b0_lesion_features")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def safe_path(path_value):
    path = Path(str(path_value))

    if path.exists():
        return path

    candidate = ROOT / path
    if candidate.exists():
        return candidate

    return path


def build_dataframe():
    frames = []

    for split in ["train", "valid", "test"]:
        df = load_aptos_split(split).copy()
        df["dataset"] = "APTOS2019"
        df["split"] = split
        frames.append(df)

    idrid_test = load_idrid_grading_split("test").copy()
    idrid_test["dataset"] = "IDRiD"
    idrid_test["split"] = "test"
    frames.append(idrid_test)

    combined = pd.concat(frames, ignore_index=True)

    if "image_exists" in combined.columns:
        combined = combined[combined["image_exists"] == True].copy()

    combined = combined[combined["label"].isin(DR_LABELS)].copy()
    combined["label"] = combined["label"].astype(int)
    combined["image_path"] = combined["image_path"].astype(str)

    return combined.reset_index(drop=True)


class FundusClassificationDataset(Dataset):
    def __init__(self, dataframe, image_size):
        self.df = dataframe.reset_index(drop=True)
        self.transform = get_classification_transforms(image_size=image_size, train=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index].to_dict()

        image_path = safe_path(row["image_path"])
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)

        return {
            "image": transformed["image"],
            "dataset": str(row["dataset"]),
            "split": str(row["split"]),
            "image_id": str(row["image_id"]),
            "label": int(row["label"]),
            "label_name": str(row["label_name"]),
            "image_path": str(image_path),
        }


def load_efficientnet(checkpoint_path, model_name, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    checkpoint_model_name = checkpoint.get("model_name", model_name)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = create_classifier(
        model_name=checkpoint_model_name,
        num_classes=5,
        pretrained=False,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint_model_name


@torch.no_grad()
def extract_efficientnet_probabilities(model, dataframe, args, device):
    dataset = FundusClassificationDataset(
        dataframe=dataframe,
        image_size=args.image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    rows = []

    for batch in tqdm(loader, desc="Extract EfficientNet probabilities"):
        images = batch["image"].to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=args.amp and device.type == "cuda",
        ):
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

        probs_np = probs.detach().cpu().numpy()
        preds_np = probs_np.argmax(axis=1)
        confidence_np = probs_np.max(axis=1)
        expected_grade_np = (probs_np * np.array(DR_LABELS).reshape(1, -1)).sum(axis=1)

        for i in range(images.size(0)):
            row = {
                "dataset": batch["dataset"][i],
                "split": batch["split"][i],
                "image_id": batch["image_id"][i],
                "label": int(batch["label"][i]),
                "label_name": batch["label_name"][i],
                "image_path": batch["image_path"][i],
                "effnet_prediction": int(preds_np[i]),
                "effnet_prediction_name": DR_LABEL_NAMES[int(preds_np[i])],
                "effnet_confidence": float(confidence_np[i]),
                "effnet_expected_grade": float(expected_grade_np[i]),
            }

            for class_id in DR_LABELS:
                row[f"effnet_prob_{class_id}"] = float(probs_np[i, class_id])

            rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


def save_confusion_matrix(y_true, y_pred, title, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=DR_LABELS)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(DR_LABELS)), DR_LABELS)
    plt.yticks(range(len(DR_LABELS)), DR_LABELS)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def get_lesion_feature_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = [
        col for col in numeric_cols
        if col not in METADATA_COLUMNS
        and col != "label"
        and not col.endswith("_lesion")
    ]

    feature_cols = sorted(feature_cols)
    return feature_cols


def make_models(seed):
    return {
        "prob_only_logistic_balanced": {
            "feature_set": "prob_only",
            "model": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        },
        "late_fusion_logistic_balanced": {
            "feature_set": "fusion",
            "model": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        },
        "late_fusion_random_forest_balanced": {
            "feature_set": "fusion",
            "model": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=500,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        },
        "late_fusion_extra_trees_balanced": {
            "feature_set": "fusion",
            "model": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            min_samples_leaf=2,
                            class_weight="balanced",
                            random_state=seed,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        },
    }


def evaluate_predictions(y_true, y_pred, model_name, split_name, output_table_dir, output_figure_dir, df_for_predictions):
    metrics = compute_metrics(y_true, y_pred)
    metrics["model"] = model_name
    metrics["split"] = split_name
    metrics["num_images"] = len(y_true)

    pred_df = df_for_predictions[
        [
            "dataset",
            "split",
            "image_id",
            "label",
            "label_name",
            "image_path",
        ]
    ].copy()

    pred_df["prediction"] = y_pred
    pred_df["prediction_name"] = [
        DR_LABEL_NAMES[int(pred)] if int(pred) in DR_LABELS else "unknown"
        for pred in y_pred
    ]

    report = classification_report(
        y_true,
        y_pred,
        labels=DR_LABELS,
        target_names=DR_LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose().reset_index().rename(
        columns={"index": "label"}
    )

    cm = confusion_matrix(y_true, y_pred, labels=DR_LABELS)

    pred_path = output_table_dir / f"{model_name}_{split_name}_predictions.csv"
    report_path = output_table_dir / f"{model_name}_{split_name}_classification_report.csv"
    cm_path = output_table_dir / f"{model_name}_{split_name}_confusion_matrix.csv"
    cm_fig_path = output_figure_dir / f"{model_name}_{split_name}_confusion_matrix.png"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(cm, index=DR_LABELS, columns=DR_LABELS).to_csv(
        cm_path,
        encoding="utf-8-sig",
    )

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        title=f"{model_name} - {split_name}",
        output_path=cm_fig_path,
    )

    return metrics


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_table_dir = ROOT / "reports" / "tables" / args.run_name
    output_figure_dir = ROOT / "reports" / "figures" / args.run_name
    checkpoint_dir = ROOT / "checkpoints" / args.run_name

    output_table_dir.mkdir(parents=True, exist_ok=True)
    output_figure_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("RetinaGuard-AI Stage 6.4: Late Fusion Classifier")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"EfficientNet checkpoint: {args.efficientnet_checkpoint}")
    print(f"Lesion features CSV: {args.lesion_features_csv}")
    print(f"Run name: {args.run_name}")
    print("=" * 100)

    image_df = build_dataframe()

    print("Input image counts:")
    print(image_df.groupby(["dataset", "split"]).size().to_string())

    model, checkpoint_model_name = load_efficientnet(
        checkpoint_path=args.efficientnet_checkpoint,
        model_name=args.model_name,
        device=device,
    )

    print(f"Loaded image model: {checkpoint_model_name}")

    effnet_df = extract_efficientnet_probabilities(
        model=model,
        dataframe=image_df,
        args=args,
        device=device,
    )

    effnet_path = output_table_dir / "efficientnet_probability_features.csv"
    effnet_df.to_csv(effnet_path, index=False, encoding="utf-8-sig")

    lesion_df = pd.read_csv(ROOT / args.lesion_features_csv)

    key_cols = ["dataset", "split", "image_id"]

    merged = effnet_df.merge(
        lesion_df,
        on=key_cols,
        how="inner",
        suffixes=("", "_lesion"),
    )

    if len(merged) != len(effnet_df):
        print()
        print("WARNING: Merge count mismatch")
        print(f"EfficientNet rows: {len(effnet_df)}")
        print(f"Merged rows:       {len(merged)}")

    merged_path = output_table_dir / "late_fusion_feature_table.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

    print()
    print("Merged counts:")
    print(merged.groupby(["dataset", "split"]).size().to_string())

    prob_cols = [f"effnet_prob_{i}" for i in DR_LABELS]
    image_meta_cols = [
        "effnet_confidence",
        "effnet_expected_grade",
    ]

    lesion_feature_cols = get_lesion_feature_columns(lesion_df)

    prob_only_cols = prob_cols + image_meta_cols
    fusion_cols = prob_only_cols + lesion_feature_cols

    feature_manifest = pd.DataFrame({
        "feature": fusion_cols,
        "source": [
            "efficientnet" if col in prob_only_cols else "lesion"
            for col in fusion_cols
        ],
    })

    feature_manifest_path = output_table_dir / "late_fusion_feature_columns.csv"
    feature_manifest.to_csv(feature_manifest_path, index=False, encoding="utf-8-sig")

    train_df = merged[(merged["dataset"] == "APTOS2019") & (merged["split"] == "train")].copy()
    valid_df = merged[(merged["dataset"] == "APTOS2019") & (merged["split"] == "valid")].copy()
    test_df = merged[(merged["dataset"] == "APTOS2019") & (merged["split"] == "test")].copy()
    idrid_test_df = merged[(merged["dataset"] == "IDRiD") & (merged["split"] == "test")].copy()

    split_map = {
        "aptos_valid": valid_df,
        "aptos_test": test_df,
        "idrid_test": idrid_test_df,
    }

    print()
    print("Fusion split counts:")
    print(f"train: {len(train_df)}")
    for split_name, split_df in split_map.items():
        print(f"{split_name}: {len(split_df)}")

    all_metrics = []

    print()
    print("=" * 100)
    print("Baseline EfficientNet raw predictions")
    print("=" * 100)

    for split_name, split_df in split_map.items():
        y_true = split_df["label"].astype(int).values
        y_pred = split_df["effnet_prediction"].astype(int).values

        metrics = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            model_name="efficientnet_raw",
            split_name=split_name,
            output_table_dir=output_table_dir,
            output_figure_dir=output_figure_dir,
            df_for_predictions=split_df,
        )

        metrics["feature_set"] = "image_only"
        all_metrics.append(metrics)

        print(
            f"{split_name}: "
            f"acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f} "
            f"qwk={metrics['qwk']:.4f}"
        )

    models = make_models(args.seed)

    best_model_name = None
    best_valid_qwk = -999.0
    best_valid_macro_f1 = -999.0

    y_train = train_df["label"].astype(int).values

    for model_name, config in models.items():
        feature_set = config["feature_set"]
        clf = config["model"]

        if feature_set == "prob_only":
            feature_cols = prob_only_cols
        else:
            feature_cols = fusion_cols

        print()
        print("=" * 100)
        print(f"Training fusion model: {model_name}")
        print(f"Feature set: {feature_set}")
        print(f"Number of features: {len(feature_cols)}")
        print("=" * 100)

        X_train = train_df[feature_cols]
        clf.fit(X_train, y_train)

        joblib.dump(clf, checkpoint_dir / f"{model_name}.joblib")

        for split_name, split_df in split_map.items():
            X = split_df[feature_cols]
            y_true = split_df["label"].astype(int).values
            y_pred = clf.predict(X)

            metrics = evaluate_predictions(
                y_true=y_true,
                y_pred=y_pred,
                model_name=model_name,
                split_name=split_name,
                output_table_dir=output_table_dir,
                output_figure_dir=output_figure_dir,
                df_for_predictions=split_df,
            )

            metrics["feature_set"] = feature_set
            all_metrics.append(metrics)

            print(
                f"{split_name}: "
                f"acc={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f} "
                f"weighted_f1={metrics['weighted_f1']:.4f} "
                f"qwk={metrics['qwk']:.4f}"
            )

            if split_name == "aptos_valid":
                if (
                    metrics["qwk"] > best_valid_qwk
                    or (
                        metrics["qwk"] == best_valid_qwk
                        and metrics["macro_f1"] > best_valid_macro_f1
                    )
                ):
                    best_valid_qwk = metrics["qwk"]
                    best_valid_macro_f1 = metrics["macro_f1"]
                    best_model_name = model_name

    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = output_table_dir / "late_fusion_metrics.csv"
    best_path = output_table_dir / "best_late_fusion_model.txt"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    best_path.write_text(
        f"best_model_name={best_model_name}\n"
        f"selection_split=aptos_valid\n"
        f"selection_metric=qwk\n"
        f"best_valid_qwk={best_valid_qwk}\n"
        f"best_valid_macro_f1={best_valid_macro_f1}\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("Metrics comparison")
    print("=" * 100)
    print(
        metrics_df[
            [
                "model",
                "feature_set",
                "split",
                "num_images",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "qwk",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 100)
    print(f"Best late-fusion model selected on APTOS valid: {best_model_name}")
    print(f"Best valid QWK: {best_valid_qwk:.4f}")
    print(f"Best valid Macro F1: {best_valid_macro_f1:.4f}")
    print("=" * 100)

    print()
    print("Saved:")
    print(effnet_path)
    print(merged_path)
    print(feature_manifest_path)
    print(metrics_path)
    print(best_path)
    print(output_figure_dir)
    print(checkpoint_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()
