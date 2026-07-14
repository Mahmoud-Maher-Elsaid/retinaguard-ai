from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_csv(path):
    path = ROOT / path
    if not path.exists():
        print(f"Missing file: {path}")
        return None
    return pd.read_csv(path)


def add_section(rows, section, finding, evidence, decision):
    rows.append({
        "section": section,
        "finding": finding,
        "evidence": evidence,
        "decision": decision,
    })


def main():
    output_dir = ROOT / "reports" / "tables" / "final_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    readme_report_path = ROOT / "reports" / "final_experiment_summary.md"

    findings = []

    # ---------------------------------------------------------------------
    # Classification experiments
    # ---------------------------------------------------------------------
    cls = read_csv(
        "reports/tables/classifier_comparison/efficientnet_b0_three_experiment_metrics_comparison.csv"
    )

    final_classification_rows = []

    if cls is not None:
        for _, row in cls.iterrows():
            final_classification_rows.append({
                "experiment_group": "EfficientNet-B0 classification",
                "model": row["experiment"],
                "split": "APTOS test",
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "weighted_f1": row["weighted_f1"],
                "qwk": row["qwk"],
                "notes": row["run_name"],
            })

        best_qwk_row = cls.sort_values("qwk", ascending=False).iloc[0]
        best_macro_row = cls.sort_values("macro_f1", ascending=False).iloc[0]

        add_section(
            findings,
            "Classification",
            "Best EfficientNet-B0 overall model is no-sampler.",
            f"Highest QWK={best_qwk_row['qwk']:.4f}, accuracy={best_qwk_row['accuracy']:.4f}, weighted_f1={best_qwk_row['weighted_f1']:.4f}.",
            "Use efficientnet_b0_aptos_100ep_bs16_no_sampler as the main image-only baseline."
        )

        add_section(
            findings,
            "Classification",
            "Label smoothing improved macro F1 slightly but reduced QWK.",
            f"Best macro_f1 experiment={best_macro_row['experiment']} with macro_f1={best_macro_row['macro_f1']:.4f}.",
            "Keep label smoothing as an ablation, not the final baseline."
        )

    # ---------------------------------------------------------------------
    # Segmentation experiments
    # ---------------------------------------------------------------------
    seg_raw = read_csv(
        "reports/tables/unet_resnet34_idrid_100ep/test_segmentation_metrics.csv"
    )
    seg_tuned = read_csv(
        "reports/tables/unet_resnet34_idrid_100ep/test_metrics_tuned_thresholds_summary.csv"
    )

    final_segmentation_rows = []

    if seg_raw is not None:
        row = seg_raw.iloc[0]
        final_segmentation_rows.append({
            "experiment_group": "U-Net segmentation",
            "model": "U-Net ResNet34 raw threshold 0.5",
            "split": "IDRiD test",
            "mean_dice": row.get("mean_dice"),
            "mean_iou": row.get("mean_iou"),
            "dice_microaneurysms": row.get("dice_microaneurysms"),
            "dice_haemorrhages": row.get("dice_haemorrhages"),
            "dice_hard_exudates": row.get("dice_hard_exudates"),
            "dice_soft_exudates": row.get("dice_soft_exudates"),
            "notes": "Single threshold 0.5",
        })

    if seg_tuned is not None:
        row = seg_tuned.iloc[0]
        final_segmentation_rows.append({
            "experiment_group": "U-Net segmentation",
            "model": "U-Net ResNet34 validation-tuned thresholds",
            "split": "IDRiD test",
            "mean_dice": row.get("mean_dice"),
            "mean_iou": row.get("mean_iou"),
            "dice_microaneurysms": row.get("dice_microaneurysms"),
            "dice_haemorrhages": row.get("dice_haemorrhages"),
            "dice_hard_exudates": row.get("dice_hard_exudates"),
            "dice_soft_exudates": row.get("dice_soft_exudates"),
            "notes": "Per-lesion thresholds selected on validation subset",
        })

        add_section(
            findings,
            "Segmentation",
            "Validation-tuned thresholds improved U-Net segmentation substantially.",
            f"IDRiD test mean Dice={row.get('mean_dice'):.4f}, hard exudates Dice={row.get('dice_hard_exudates'):.4f}, soft exudates Dice={row.get('dice_soft_exudates'):.4f}.",
            "Use tuned-threshold segmentation results in README and qualitative figures."
        )

        add_section(
            findings,
            "Segmentation",
            "Microaneurysm segmentation remains the weakest lesion channel.",
            f"Microaneurysm Dice={row.get('dice_microaneurysms'):.4f}.",
            "Report this honestly as a limitation and future improvement target."
        )

    # ---------------------------------------------------------------------
    # Lesion feature classifier
    # ---------------------------------------------------------------------
    lesion_clf = read_csv(
        "reports/tables/lesion_feature_classifier_aptos_train/lesion_feature_classifier_metrics.csv"
    )

    final_lesion_classifier_rows = []

    if lesion_clf is not None:
        for _, row in lesion_clf.iterrows():
            final_lesion_classifier_rows.append({
                "experiment_group": "Lesion-feature-only classifier",
                "model": row["model"],
                "split": row["split"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "weighted_f1": row["weighted_f1"],
                "qwk": row["qwk"],
                "notes": "Uses U-Net lesion statistics only",
            })

        aptos_test = lesion_clf[lesion_clf["split"] == "aptos_test"].copy()
        best_lesion_test = aptos_test.sort_values("qwk", ascending=False).iloc[0]

        add_section(
            findings,
            "Lesion features",
            "Lesion features alone contain useful disease signal but are weaker than EfficientNet.",
            f"Best lesion-feature-only APTOS test QWK={best_lesion_test['qwk']:.4f} using {best_lesion_test['model']}.",
            "Use lesion features as interpretable auxiliary features, not as a replacement for image models."
        )

    # ---------------------------------------------------------------------
    # Late fusion
    # ---------------------------------------------------------------------
    fusion = read_csv(
        "reports/tables/late_fusion_effnet_b0_lesion_features/late_fusion_metrics.csv"
    )

    final_fusion_rows = []

    if fusion is not None:
        for _, row in fusion.iterrows():
            final_fusion_rows.append({
                "experiment_group": "Late fusion",
                "model": row["model"],
                "feature_set": row["feature_set"],
                "split": row["split"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "weighted_f1": row["weighted_f1"],
                "qwk": row["qwk"],
                "notes": "EfficientNet probabilities with or without lesion features",
            })

        aptos_test_fusion = fusion[fusion["split"] == "aptos_test"].copy()
        best_fusion_test = aptos_test_fusion.sort_values("qwk", ascending=False).iloc[0]
        best_macro_test = aptos_test_fusion.sort_values("macro_f1", ascending=False).iloc[0]

        add_section(
            findings,
            "Late fusion",
            "Probability-only meta-classifier achieved the highest APTOS test QWK.",
            f"Best APTOS test QWK={best_fusion_test['qwk']:.4f} using {best_fusion_test['model']}.",
            "Treat this as calibration/meta-classification, not lesion-based improvement."
        )

        add_section(
            findings,
            "Late fusion",
            "Lesion fusion improved some metrics but did not consistently dominate QWK.",
            f"Best APTOS test Macro F1={best_macro_test['macro_f1']:.4f} using {best_macro_test['model']}.",
            "Report fusion results as mixed: useful for interpretability and some metric gains, but image-only EfficientNet remains very strong."
        )

    # ---------------------------------------------------------------------
    # Lesion feature QA correlations
    # ---------------------------------------------------------------------
    corr = read_csv(
        "reports/tables/lesion_feature_qa/unet_resnet34_tuned_thresholds_feature_grade_correlations.csv"
    )

    final_correlation_rows = []

    if corr is not None:
        selected_features = [
            "total_lesion_union_area_ratio",
            "hard_exudates_area_ratio",
            "haemorrhages_area_ratio",
            "microaneurysms_area_ratio",
            "soft_exudates_area_ratio",
            "lesion_presence_count",
        ]

        key_corr = corr[
            (corr["dataset"].isin(["ALL", "APTOS2019", "IDRiD"])) &
            (corr["split"] == "ALL") &
            (corr["feature"].isin(selected_features))
        ].copy()

        key_corr = key_corr.sort_values(
            ["dataset", "spearman_corr_with_grade"],
            ascending=[True, False],
        )

        final_correlation_rows = key_corr.to_dict("records")

        all_total = key_corr[
            (key_corr["dataset"] == "ALL") &
            (key_corr["feature"] == "total_lesion_union_area_ratio")
        ]

        if len(all_total) > 0:
            row = all_total.iloc[0]
            add_section(
                findings,
                "Lesion feature QA",
                "Total predicted lesion area strongly correlates with DR grade.",
                f"ALL Spearman correlation={row['spearman_corr_with_grade']:.4f}.",
                "Use this as the main evidence that U-Net-derived lesion features are medically meaningful."
            )

    # ---------------------------------------------------------------------
    # Save final tables
    # ---------------------------------------------------------------------
    final_classification_df = pd.DataFrame(final_classification_rows)
    final_segmentation_df = pd.DataFrame(final_segmentation_rows)
    final_lesion_classifier_df = pd.DataFrame(final_lesion_classifier_rows)
    final_fusion_df = pd.DataFrame(final_fusion_rows)
    final_correlation_df = pd.DataFrame(final_correlation_rows)
    findings_df = pd.DataFrame(findings)

    final_classification_df.to_csv(output_dir / "final_classification_results.csv", index=False, encoding="utf-8-sig")
    final_segmentation_df.to_csv(output_dir / "final_segmentation_results.csv", index=False, encoding="utf-8-sig")
    final_lesion_classifier_df.to_csv(output_dir / "final_lesion_feature_classifier_results.csv", index=False, encoding="utf-8-sig")
    final_fusion_df.to_csv(output_dir / "final_late_fusion_results.csv", index=False, encoding="utf-8-sig")
    final_correlation_df.to_csv(output_dir / "final_lesion_feature_correlations.csv", index=False, encoding="utf-8-sig")
    findings_df.to_csv(output_dir / "final_key_findings.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------------
    # Save markdown report
    # ---------------------------------------------------------------------
    lines = []
    lines.append("# RetinaGuard-AI Final Experiment Summary")
    lines.append("")
    lines.append("## Main Classification Result")
    if cls is not None:
        best = cls.sort_values("qwk", ascending=False).iloc[0]
        lines.append(
            f"- Best EfficientNet-B0 baseline: `{best['experiment']}` with "
            f"APTOS test QWK `{best['qwk']:.4f}`, accuracy `{best['accuracy']:.4f}`, "
            f"macro F1 `{best['macro_f1']:.4f}`, weighted F1 `{best['weighted_f1']:.4f}`."
        )

    lines.append("")
    lines.append("## Main Segmentation Result")
    if seg_tuned is not None:
        row = seg_tuned.iloc[0]
        lines.append(
            f"- Tuned-threshold U-Net ResNet34 achieved IDRiD test mean Dice `{row.get('mean_dice'):.4f}` "
            f"and mean IoU `{row.get('mean_iou'):.4f}`."
        )
        lines.append(
            f"- Per-lesion Dice: MA `{row.get('dice_microaneurysms'):.4f}`, "
            f"Haemorrhages `{row.get('dice_haemorrhages'):.4f}`, "
            f"Hard Exudates `{row.get('dice_hard_exudates'):.4f}`, "
            f"Soft Exudates `{row.get('dice_soft_exudates'):.4f}`."
        )

    lines.append("")
    lines.append("## Lesion Feature Findings")
    lines.append("- U-Net-derived lesion features showed strong correlation with DR grade.")
    lines.append("- Lesion-feature-only classifiers were weaker than image-based EfficientNet, but confirmed that lesion statistics contain useful disease signal.")

    lines.append("")
    lines.append("## Late Fusion Finding")
    if fusion is not None:
        best_test = fusion[fusion["split"] == "aptos_test"].sort_values("qwk", ascending=False).iloc[0]
        lines.append(
            f"- Best APTOS test late-fusion/meta-classifier result: `{best_test['model']}` "
            f"with QWK `{best_test['qwk']:.4f}`, accuracy `{best_test['accuracy']:.4f}`, "
            f"macro F1 `{best_test['macro_f1']:.4f}`."
        )
        lines.append("- Lesion fusion produced mixed gains, so the final claim should be conservative and honest.")

    lines.append("")
    lines.append("## Final Project Claim")
    lines.append(
        "RetinaGuard-AI provides an end-to-end diabetic retinopathy pipeline combining image-based grading, lesion segmentation, lesion-derived statistical analysis, and late-fusion experiments. "
        "The strongest grading performance comes from EfficientNet-based image classification, while U-Net lesion features provide interpretable medical evidence and useful auxiliary signals."
    )

    readme_report_path.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 100)
    print("RetinaGuard-AI Stage 7: Final Experiment Summary")
    print("=" * 100)
    print("Saved final summary tables:")
    print(output_dir / "final_classification_results.csv")
    print(output_dir / "final_segmentation_results.csv")
    print(output_dir / "final_lesion_feature_classifier_results.csv")
    print(output_dir / "final_late_fusion_results.csv")
    print(output_dir / "final_lesion_feature_correlations.csv")
    print(output_dir / "final_key_findings.csv")
    print()
    print("Saved markdown summary:")
    print(readme_report_path)
    print()
    print("Key findings:")
    print(findings_df.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
