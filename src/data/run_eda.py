from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset_registry import (
    DR_CLASS_NAMES,
    load_aptos_all,
    load_idrid_grading_all,
    classification_summary,
    idrid_segmentation_mask_summary,
)

REPORT_TABLES = ROOT / "reports" / "tables"
REPORT_FIGURES = ROOT / "reports" / "figures"

REPORT_TABLES.mkdir(parents=True, exist_ok=True)
REPORT_FIGURES.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(summary: pd.DataFrame, dataset: str, output_path: Path) -> None:
    subset = summary[summary["dataset"] == dataset].copy()

    if subset.empty:
        print(f"No rows found for {dataset}. Skipping plot.")
        return

    pivot = (
        subset.pivot_table(
            index="label",
            columns="split",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
    )

    ax = pivot.plot(kind="bar", figsize=(10, 5))

    labels = [
        f"{int(label)} - {DR_CLASS_NAMES.get(int(label), 'Unknown')}"
        for label in pivot.index
    ]

    ax.set_title(f"{dataset} Diabetic Retinopathy Class Distribution")
    ax.set_xlabel("DR Grade")
    ax.set_ylabel("Number of Images")
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(title="Split")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_segmentation_mask_counts(seg_summary: pd.DataFrame, output_path: Path) -> None:
    if seg_summary.empty:
        print("No segmentation summary found. Skipping segmentation plot.")
        return

    pivot = (
        seg_summary.pivot_table(
            index="mask_folder",
            columns="split",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
    )

    ax = pivot.plot(kind="bar", figsize=(11, 5))

    ax.set_title("IDRiD Segmentation Mask Counts")
    ax.set_xlabel("Lesion / Mask Folder")
    ax.set_ylabel("Number of Masks")
    ax.set_xticklabels(pivot.index, rotation=25, ha="right")
    ax.legend(title="Split")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    print("=" * 80)
    print("RetinaGuard-AI Stage 2: Dataset Registry + EDA")
    print("=" * 80)

    aptos_df = load_aptos_all()
    idrid_df = load_idrid_grading_all()

    all_cls_df = pd.concat([aptos_df, idrid_df], ignore_index=True)

    missing_images = all_cls_df[~all_cls_df["image_exists"]]

    if len(missing_images) > 0:
        print("WARNING: Missing images detected:")
        print(missing_images[["dataset", "split", "image_id"]].head(20).to_string(index=False))
    else:
        print("All classification labels are matched with images.")

    cls_summary = classification_summary(all_cls_df)
    seg_summary = idrid_segmentation_mask_summary()

    cls_summary_path = REPORT_TABLES / "classification_class_distribution.csv"
    seg_summary_path = REPORT_TABLES / "idrid_segmentation_mask_summary.csv"

    cls_summary.to_csv(cls_summary_path, index=False, encoding="utf-8-sig")
    seg_summary.to_csv(seg_summary_path, index=False, encoding="utf-8-sig")

    plot_class_distribution(
        cls_summary,
        "APTOS2019",
        REPORT_FIGURES / "aptos2019_class_distribution.png",
    )

    plot_class_distribution(
        cls_summary,
        "IDRiD",
        REPORT_FIGURES / "idrid_class_distribution.png",
    )

    plot_segmentation_mask_counts(
        seg_summary,
        REPORT_FIGURES / "idrid_segmentation_mask_counts.png",
    )

    print()
    print("Classification summary:")
    print(cls_summary.to_string(index=False))

    print()
    print("Segmentation mask summary:")
    print(seg_summary[["dataset", "split", "mask_folder", "count"]].to_string(index=False))

    print()
    print("Saved tables:")
    print(cls_summary_path)
    print(seg_summary_path)

    print()
    print("Saved figures:")
    print(REPORT_FIGURES / "aptos2019_class_distribution.png")
    print(REPORT_FIGURES / "idrid_class_distribution.png")
    print(REPORT_FIGURES / "idrid_segmentation_mask_counts.png")

    print("=" * 80)
    print("Stage 2 EDA completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
