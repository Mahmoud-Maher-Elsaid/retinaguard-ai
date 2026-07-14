import argparse
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features-csv",
        type=str,
        default="reports/tables/lesion_features/unet_resnet34_tuned_thresholds_lesion_features_all.csv",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="lesion_feature_classifier_aptos_train",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


def get_feature_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = [
        col for col in numeric_cols
        if col not in METADATA_COLUMNS
        and col != "label"
    ]

    feature_cols = sorted(feature_cols)

    return feature_cols


def split_data(df):
    aptos_train = df[(df["dataset"] == "APTOS2019") & (df["split"] == "train")].copy()
    aptos_valid = df[(df["dataset"] == "APTOS2019") & (df["split"] == "valid")].copy()
    aptos_test = df[(df["dataset"] == "APTOS2019") & (df["split"] == "test")].copy()
    idrid_test = df[(df["dataset"] == "IDRiD") & (df["split"] == "test")].copy()

    return {
        "aptos_train": aptos_train,
        "aptos_valid": aptos_valid,
        "aptos_test": aptos_test,
        "idrid_test": idrid_test,
    }


def make_models(seed):
    models = {
        "logistic_regression_balanced": Pipeline(
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
        "random_forest_balanced": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "extra_trees_balanced": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    return models


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


def evaluate_model(model, df, feature_cols, split_name, output_table_dir, output_figure_dir, model_name):
    X = df[feature_cols]
    y_true = df["label"].astype(int).values

    y_pred = model.predict(X)

    metrics = compute_metrics(y_true, y_pred)
    metrics["split"] = split_name
    metrics["model"] = model_name
    metrics["num_images"] = len(df)

    pred_df = df[
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

    pred_path = output_table_dir / f"{model_name}_{split_name}_predictions.csv"
    report_path = output_table_dir / f"{model_name}_{split_name}_classification_report.csv"
    cm_path = output_table_dir / f"{model_name}_{split_name}_confusion_matrix.csv"
    cm_fig_path = output_figure_dir / f"{model_name}_{split_name}_confusion_matrix.png"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=DR_LABELS)
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

    features_path = ROOT / args.features_csv

    output_table_dir = ROOT / "reports" / "tables" / args.run_name
    output_figure_dir = ROOT / "reports" / "figures" / args.run_name
    checkpoint_dir = ROOT / "checkpoints" / args.run_name

    output_table_dir.mkdir(parents=True, exist_ok=True)
    output_figure_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("RetinaGuard-AI Stage 6.3: Lesion-Feature-Only Classifier")
    print("=" * 100)
    print(f"Features CSV: {features_path}")
    print(f"Run name: {args.run_name}")
    print("=" * 100)

    df = pd.read_csv(features_path)
    df = df[df["label"].isin(DR_LABELS)].copy()
    df["label"] = df["label"].astype(int)

    splits = split_data(df)

    print("Input counts:")
    for name, split_df in splits.items():
        print(f"{name}: {len(split_df)}")

    feature_cols = get_feature_columns(df)

    print()
    print(f"Number of lesion features: {len(feature_cols)}")
    for col in feature_cols:
        print(f"- {col}")

    X_train = splits["aptos_train"][feature_cols]
    y_train = splits["aptos_train"]["label"].astype(int).values

    models = make_models(seed=args.seed)

    all_metrics = []

    best_model_name = None
    best_valid_qwk = -999.0
    best_valid_macro_f1 = -999.0

    for model_name, model in models.items():
        print()
        print("=" * 100)
        print(f"Training model: {model_name}")
        print("=" * 100)

        model.fit(X_train, y_train)

        joblib.dump(model, checkpoint_dir / f"{model_name}.joblib")

        for split_name in ["aptos_valid", "aptos_test", "idrid_test"]:
            metrics = evaluate_model(
                model=model,
                df=splits[split_name],
                feature_cols=feature_cols,
                split_name=split_name,
                output_table_dir=output_table_dir,
                output_figure_dir=output_figure_dir,
                model_name=model_name,
            )

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

    metrics_path = output_table_dir / "lesion_feature_classifier_metrics.csv"
    feature_path = output_table_dir / "lesion_feature_columns.csv"
    best_path = output_table_dir / "best_lesion_feature_classifier.txt"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": feature_cols}).to_csv(
        feature_path,
        index=False,
        encoding="utf-8-sig",
    )

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
    print(f"Best model selected on APTOS valid: {best_model_name}")
    print(f"Best valid QWK: {best_valid_qwk:.4f}")
    print(f"Best valid Macro F1: {best_valid_macro_f1:.4f}")
    print("=" * 100)

    print()
    print("Saved:")
    print(metrics_path)
    print(feature_path)
    print(best_path)
    print(output_figure_dir)
    print(checkpoint_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()
