import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


DR_LABEL_ORDER = [0, 1, 2, 3, 4]
DR_LABEL_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

KEY_FEATURES = [
    "total_lesion_union_area_ratio",
    "total_lesion_sum_area_ratio",
    "microaneurysms_area_ratio",
    "haemorrhages_area_ratio",
    "hard_exudates_area_ratio",
    "soft_exudates_area_ratio",
    "microaneurysms_prob_mean",
    "haemorrhages_prob_mean",
    "hard_exudates_prob_mean",
    "soft_exudates_prob_mean",
    "microaneurysms_component_count",
    "haemorrhages_component_count",
    "hard_exudates_component_count",
    "soft_exudates_component_count",
    "lesion_presence_count",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-csv",
        type=str,
        default="reports/tables/lesion_features/unet_resnet34_tuned_thresholds_lesion_features_all.csv",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="unet_resnet34_tuned_thresholds",
    )
    return parser.parse_args()


def clean_dataframe(df):
    df = df.copy()

    df = df[df["label"].isin(DR_LABEL_ORDER)].copy()
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map(DR_LABEL_NAMES)

    return df


def compute_grade_summary(df, feature_cols):
    agg = {}

    for feature in feature_cols:
        agg[feature] = ["mean", "std", "median", "min", "max"]

    summary = (
        df.groupby(["dataset", "split", "label", "label_name"])
        .agg(agg)
        .reset_index()
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    counts = (
        df.groupby(["dataset", "split", "label", "label_name"])
        .size()
        .reset_index(name="num_images")
    )

    summary = counts.merge(
        summary,
        on=["dataset", "split", "label", "label_name"],
        how="left",
    )

    return summary


def compute_feature_correlations(df, feature_cols):
    rows = []

    groups = []

    groups.append(("ALL", "ALL", df))

    for dataset_name, dataset_df in df.groupby("dataset"):
        groups.append((dataset_name, "ALL", dataset_df))

    for (dataset_name, split_name), group in df.groupby(["dataset", "split"]):
        groups.append((dataset_name, split_name, group))

    for dataset_name, split_name, group in groups:
        if len(group) < 5:
            continue

        for feature in feature_cols:
            if feature not in group.columns:
                continue

            values = group[feature]

            if values.nunique(dropna=True) <= 1:
                spearman_corr = np.nan
                pearson_corr = np.nan
            else:
                spearman_corr = group[["label", feature]].corr(method="spearman").iloc[0, 1]
                pearson_corr = group[["label", feature]].corr(method="pearson").iloc[0, 1]

            grade_means = group.groupby("label")[feature].mean()

            grade_0 = float(grade_means.get(0, np.nan))
            grade_4 = float(grade_means.get(4, np.nan))
            grade_4_minus_0 = grade_4 - grade_0 if not np.isnan(grade_0) and not np.isnan(grade_4) else np.nan

            rows.append({
                "dataset": dataset_name,
                "split": split_name,
                "feature": feature,
                "num_images": len(group),
                "spearman_corr_with_grade": spearman_corr,
                "pearson_corr_with_grade": pearson_corr,
                "mean_grade_0": grade_0,
                "mean_grade_4": grade_4,
                "grade_4_minus_grade_0": grade_4_minus_0,
            })

    return pd.DataFrame(rows)


def save_mean_feature_plots(df, feature_cols, output_dir, output_name):
    plot_features = [
        "total_lesion_union_area_ratio",
        "haemorrhages_area_ratio",
        "hard_exudates_area_ratio",
        "soft_exudates_area_ratio",
        "microaneurysms_area_ratio",
    ]

    plot_features = [f for f in plot_features if f in feature_cols]

    for dataset_name, dataset_df in df.groupby("dataset"):
        mean_by_grade = (
            dataset_df.groupby(["label", "label_name"])[plot_features]
            .mean()
            .reset_index()
            .sort_values("label")
        )

        for feature in plot_features:
            plt.figure(figsize=(8, 5))

            x = mean_by_grade["label"].astype(str) + " - " + mean_by_grade["label_name"]
            y = mean_by_grade[feature]

            plt.bar(x, y)
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("Mean area ratio")
            plt.title(f"{dataset_name}: {feature} by DR grade")
            plt.tight_layout()

            safe_feature = feature.replace("/", "_").replace(" ", "_")
            safe_dataset = str(dataset_name).lower().replace(" ", "_")

            out_path = output_dir / f"{output_name}_{safe_dataset}_{safe_feature}_by_grade.png"
            plt.savefig(out_path, dpi=200)
            plt.close()


def save_boxplots(df, feature_cols, output_dir, output_name):
    box_features = [
        "total_lesion_union_area_ratio",
        "haemorrhages_area_ratio",
        "hard_exudates_area_ratio",
        "soft_exudates_area_ratio",
    ]

    box_features = [f for f in box_features if f in feature_cols]

    for dataset_name, dataset_df in df.groupby("dataset"):
        for feature in box_features:
            data = []
            labels = []

            for label in DR_LABEL_ORDER:
                values = dataset_df[dataset_df["label"] == label][feature].dropna().values

                if len(values) > 0:
                    data.append(values)
                    labels.append(f"{label}\n{DR_LABEL_NAMES[label]}")

            if not data:
                continue

            plt.figure(figsize=(8, 5))
            try:
                plt.boxplot(data, tick_labels=labels, showfliers=False)
            except TypeError:
                plt.boxplot(data, showfliers=False)
                plt.xticks(range(1, len(labels) + 1), labels)
            plt.ylabel(feature)
            plt.title(f"{dataset_name}: {feature} distribution by DR grade")
            plt.tight_layout()

            safe_feature = feature.replace("/", "_").replace(" ", "_")
            safe_dataset = str(dataset_name).lower().replace(" ", "_")

            out_path = output_dir / f"{output_name}_{safe_dataset}_{safe_feature}_boxplot_by_grade.png"
            plt.savefig(out_path, dpi=200)
            plt.close()


def main():
    args = parse_args()

    features_path = ROOT / args.features_csv
    output_table_dir = ROOT / "reports" / "tables" / "lesion_feature_qa"
    output_figure_dir = ROOT / "reports" / "figures" / "lesion_feature_qa"

    output_table_dir.mkdir(parents=True, exist_ok=True)
    output_figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("RetinaGuard-AI Stage 6.2: Lesion Feature QA")
    print("=" * 100)
    print(f"Features CSV: {features_path}")

    df = pd.read_csv(features_path)
    df = clean_dataframe(df)

    feature_cols = [col for col in KEY_FEATURES if col in df.columns]

    print()
    print("Input counts:")
    print(df.groupby(["dataset", "split"]).size().to_string())

    print()
    print("Features used:")
    for col in feature_cols:
        print(f"- {col}")

    grade_summary = compute_grade_summary(df, feature_cols)
    correlations = compute_feature_correlations(df, feature_cols)

    grade_summary_path = output_table_dir / f"{args.output_name}_grade_feature_summary.csv"
    correlations_path = output_table_dir / f"{args.output_name}_feature_grade_correlations.csv"

    grade_summary.to_csv(grade_summary_path, index=False, encoding="utf-8-sig")
    correlations.to_csv(correlations_path, index=False, encoding="utf-8-sig")

    save_mean_feature_plots(
        df=df,
        feature_cols=feature_cols,
        output_dir=output_figure_dir,
        output_name=args.output_name,
    )

    save_boxplots(
        df=df,
        feature_cols=feature_cols,
        output_dir=output_figure_dir,
        output_name=args.output_name,
    )

    important = correlations[
        (correlations["dataset"].isin(["APTOS2019", "IDRiD", "ALL"])) &
        (correlations["split"] == "ALL") &
        (correlations["feature"].isin([
            "total_lesion_union_area_ratio",
            "haemorrhages_area_ratio",
            "hard_exudates_area_ratio",
            "soft_exudates_area_ratio",
            "microaneurysms_area_ratio",
            "lesion_presence_count",
        ]))
    ].copy()

    important = important.sort_values(
        ["dataset", "spearman_corr_with_grade"],
        ascending=[True, False],
    )

    print()
    print("=" * 100)
    print("Key Feature Correlations With DR Grade")
    print("=" * 100)
    print(
        important[
            [
                "dataset",
                "split",
                "feature",
                "num_images",
                "spearman_corr_with_grade",
                "pearson_corr_with_grade",
                "mean_grade_0",
                "mean_grade_4",
                "grade_4_minus_grade_0",
            ]
        ].to_string(index=False)
    )

    print()
    print("Saved:")
    print(grade_summary_path)
    print(correlations_path)
    print(output_figure_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()
