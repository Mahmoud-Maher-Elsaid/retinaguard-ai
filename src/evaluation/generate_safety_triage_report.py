import argparse
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


DR_LABEL_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

PROB_COLS = [f"effnet_prob_{i}" for i in range(5)]

LESION_AREA_COLS = [
    "microaneurysms_area_ratio",
    "haemorrhages_area_ratio",
    "hard_exudates_area_ratio",
    "soft_exudates_area_ratio",
]

LESION_DISPLAY_NAMES = {
    "microaneurysms_area_ratio": "Microaneurysms",
    "haemorrhages_area_ratio": "Haemorrhages",
    "hard_exudates_area_ratio": "Hard Exudates",
    "soft_exudates_area_ratio": "Soft Exudates",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature-table",
        type=str,
        default="reports/tables/late_fusion_effnet_b0_lesion_features/late_fusion_feature_table.csv",
    )
    parser.add_argument("--run-name", type=str, default="retinaguard_safety_gate")
    parser.add_argument("--panel-dataset", type=str, default="APTOS2019")
    parser.add_argument("--panel-split", type=str, default="test")
    parser.add_argument("--num-panels", type=int, default=15)
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


def normalized_entropy(probabilities):
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probs))
    return float(entropy / max_entropy)


def top_margin(probabilities):
    probs = np.sort(np.asarray(probabilities, dtype=np.float64))[::-1]
    return float(probs[0] - probs[1])


def read_image_for_quality(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    return image


def compute_image_quality(image_bgr):
    if image_bgr is None:
        return {
            "image_quality_status": "unreadable",
            "quality_score": 0.0,
            "blur_score": 0.0,
            "brightness_score": 0.0,
            "contrast_score": 0.0,
            "brightness_mean": 0.0,
            "contrast_std": 0.0,
        }

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    blur_score_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    blur_score = float(np.clip(blur_score_raw / 350.0, 0.0, 1.0))

    brightness_centered = 1.0 - abs(brightness_mean - 115.0) / 115.0
    brightness_score = float(np.clip(brightness_centered, 0.0, 1.0))

    contrast_score = float(np.clip(contrast_std / 65.0, 0.0, 1.0))

    quality_score = float(
        0.45 * blur_score +
        0.30 * brightness_score +
        0.25 * contrast_score
    )

    if quality_score >= 0.65:
        status = "good"
    elif quality_score >= 0.45:
        status = "acceptable"
    else:
        status = "poor"

    return {
        "image_quality_status": status,
        "quality_score": quality_score,
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "brightness_mean": brightness_mean,
        "contrast_std": contrast_std,
    }


def get_probability_info(row):
    probs = np.array([float(row[col]) for col in PROB_COLS], dtype=np.float64)

    pred = int(np.argmax(probs))
    confidence = float(np.max(probs))
    entropy = normalized_entropy(probs)
    margin = top_margin(probs)

    return probs, pred, confidence, entropy, margin


def lesion_burden_info(row, train_reference):
    lesion_values = {}

    for col in LESION_AREA_COLS:
        value = float(row.get(col, 0.0))
        lesion_values[col] = value

    if "total_lesion_union_area_ratio" in row.index:
        total_burden = float(row["total_lesion_union_area_ratio"])
    else:
        total_burden = float(sum(lesion_values.values()))

    if train_reference is not None and len(train_reference) > 0:
        percentile = float((train_reference <= total_burden).mean())
    else:
        percentile = 0.0

    dominant_col = max(lesion_values, key=lesion_values.get)
    dominant_value = lesion_values[dominant_col]

    if total_burden >= 0.025:
        level = "very_high"
    elif total_burden >= 0.012:
        level = "high"
    elif total_burden >= 0.004:
        level = "medium"
    elif total_burden >= 0.001:
        level = "low"
    else:
        level = "very_low"

    return {
        "total_lesion_burden": total_burden,
        "lesion_burden_percentile_vs_train": percentile,
        "dominant_lesion_type": LESION_DISPLAY_NAMES[dominant_col],
        "dominant_lesion_area_ratio": dominant_value,
        "lesion_evidence_level": level,
    }


def lesion_grade_consistency(pred_label, lesion_level, total_burden):
    if pred_label == 0:
        if lesion_level in ["very_low", "low"]:
            return "consistent"
        if lesion_level == "medium":
            return "borderline"
        return "inconsistent"

    if pred_label == 1:
        if lesion_level in ["low", "medium"]:
            return "consistent"
        if lesion_level in ["very_low", "high"]:
            return "borderline"
        return "inconsistent"

    if pred_label == 2:
        if lesion_level in ["medium", "high"]:
            return "consistent"
        if lesion_level in ["low", "very_high"]:
            return "borderline"
        return "inconsistent"

    if pred_label in [3, 4]:
        if lesion_level in ["high", "very_high"]:
            return "consistent"
        if lesion_level == "medium":
            return "borderline"
        return "inconsistent"

    return "unknown"


def uncertainty_level(confidence, entropy, margin):
    if confidence >= 0.85 and entropy <= 0.35 and margin >= 0.45:
        return "low"

    if confidence >= 0.65 and entropy <= 0.60 and margin >= 0.20:
        return "medium"

    return "high"


def compute_risk_score(pred_label, confidence, entropy, margin, quality_score, lesion_percentile):
    severity_component = pred_label / 4.0
    uncertainty_component = entropy
    low_margin_component = 1.0 - np.clip(margin, 0.0, 1.0)
    quality_risk = 1.0 - quality_score
    lesion_component = lesion_percentile

    risk = (
        0.35 * severity_component +
        0.20 * uncertainty_component +
        0.15 * low_margin_component +
        0.15 * quality_risk +
        0.15 * lesion_component
    )

    return float(np.clip(risk, 0.0, 1.0))


def triage_decision(pred_label, uncertainty, quality_status, consistency, lesion_level, confidence, risk_score):
    if quality_status == "unreadable":
        return "manual_review_required"

    if quality_status == "poor":
        return "manual_review_required"

    if uncertainty == "high":
        return "manual_review_required"

    if consistency == "inconsistent":
        return "manual_review_required"

    if pred_label >= 3:
        return "urgent_referral"

    if pred_label == 2:
        if lesion_level in ["high", "very_high"] or risk_score >= 0.55:
            return "urgent_referral"
        return "routine_referral"

    if pred_label == 1:
        return "follow_up_recommended"

    if pred_label == 0:
        if confidence >= 0.85 and lesion_level in ["very_low", "low"] and uncertainty == "low":
            return "safe_negative_prediction"
        return "low_risk_follow_up"

    return "manual_review_required"


def make_case_panel(row, output_path):
    image_path = safe_path(row["image_path"])
    image = cv2.imread(str(image_path))

    if image is None:
        return False

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    probs = [float(row[col]) for col in PROB_COLS]
    prob_labels = [DR_LABEL_NAMES[i] for i in range(5)]

    lesion_values = [
        float(row.get("microaneurysms_area_ratio", 0.0)),
        float(row.get("haemorrhages_area_ratio", 0.0)),
        float(row.get("hard_exudates_area_ratio", 0.0)),
        float(row.get("soft_exudates_area_ratio", 0.0)),
    ]
    lesion_labels = ["MA", "HE", "EX", "SE"]

    fig = plt.figure(figsize=(15, 9))

    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(image)
    ax1.axis("off")
    ax1.set_title(
        f"Original Fundus Image\n"
        f"True: {int(row['label'])} - {row['label_name']}"
    )

    ax2 = plt.subplot(2, 2, 2)
    y_pos = np.arange(len(prob_labels))
    ax2.barh(y_pos, probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(prob_labels)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Probability")
    ax2.set_title(
        f"EfficientNet Prediction\n"
        f"Pred: {int(row['safety_pred_label'])} - {row['safety_pred_label_name']}"
    )

    ax3 = plt.subplot(2, 2, 3)
    ax3.bar(lesion_labels, lesion_values)
    ax3.set_ylabel("Area Ratio")
    ax3.set_title(
        f"U-Net Lesion Evidence\n"
        f"Burden: {row['total_lesion_burden']:.5f} | Level: {row['lesion_evidence_level']}"
    )

    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")

    text = (
        f"RetinaGuard Safety Gate\n\n"
        f"Decision: {row['triage_decision']}\n"
        f"Risk score: {row['risk_score']:.3f}\n\n"
        f"Confidence: {row['confidence']:.3f}\n"
        f"Uncertainty: {row['uncertainty_level']}\n"
        f"Entropy: {row['entropy']:.3f}\n"
        f"Top-2 margin: {row['top2_margin']:.3f}\n\n"
        f"Image quality: {row['image_quality_status']}\n"
        f"Quality score: {row['quality_score']:.3f}\n\n"
        f"Lesion consistency: {row['lesion_grade_consistency']}\n"
        f"Dominant lesion: {row['dominant_lesion_type']}\n"
    )

    ax4.text(
        0.0,
        1.0,
        text,
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return True


def main():
    args = parse_args()

    output_table_dir = ROOT / "reports" / "tables" / args.run_name
    output_figure_dir = ROOT / "reports" / "figures" / args.run_name
    panel_dir = output_figure_dir / "sample_triage_panels"

    output_table_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    feature_path = ROOT / args.feature_table

    print("=" * 100)
    print("RetinaGuard-AI Stage 8: Safety Gate / Uncertainty-Aware Triage")
    print("=" * 100)
    print(f"Feature table: {feature_path}")
    print(f"Run name: {args.run_name}")
    print(f"Output tables: {output_table_dir}")
    print(f"Output figures: {output_figure_dir}")
    print("=" * 100)

    df = pd.read_csv(feature_path)

    missing_probs = [col for col in PROB_COLS if col not in df.columns]
    if missing_probs:
        raise ValueError(f"Missing probability columns: {missing_probs}")

    train_ref = df[
        (df["dataset"] == "APTOS2019") &
        (df["split"] == "train")
    ]["total_lesion_union_area_ratio"].astype(float)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building safety triage report"):
        image_path = safe_path(row["image_path"])
        image_bgr = read_image_for_quality(image_path)

        quality = compute_image_quality(image_bgr)

        probs, pred, confidence, entropy, margin = get_probability_info(row)
        lesion_info = lesion_burden_info(row, train_ref)

        consistency = lesion_grade_consistency(
            pred_label=pred,
            lesion_level=lesion_info["lesion_evidence_level"],
            total_burden=lesion_info["total_lesion_burden"],
        )

        unc_level = uncertainty_level(
            confidence=confidence,
            entropy=entropy,
            margin=margin,
        )

        risk_score = compute_risk_score(
            pred_label=pred,
            confidence=confidence,
            entropy=entropy,
            margin=margin,
            quality_score=quality["quality_score"],
            lesion_percentile=lesion_info["lesion_burden_percentile_vs_train"],
        )

        decision = triage_decision(
            pred_label=pred,
            uncertainty=unc_level,
            quality_status=quality["image_quality_status"],
            consistency=consistency,
            lesion_level=lesion_info["lesion_evidence_level"],
            confidence=confidence,
            risk_score=risk_score,
        )

        out = row.to_dict()

        out.update({
            "safety_pred_label": pred,
            "safety_pred_label_name": DR_LABEL_NAMES[pred],
            "confidence": confidence,
            "entropy": entropy,
            "top2_margin": margin,
            "uncertainty_level": unc_level,
            "lesion_grade_consistency": consistency,
            "risk_score": risk_score,
            "triage_decision": decision,
        })

        out.update(quality)
        out.update(lesion_info)

        rows.append(out)

    report_df = pd.DataFrame(rows)

    report_path = output_table_dir / "safety_triage_report.csv"
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    summary_by_split = (
        report_df
        .groupby(["dataset", "split", "triage_decision"])
        .size()
        .reset_index(name="count")
    )

    summary_by_split_path = output_table_dir / "safety_triage_summary_by_split.csv"
    summary_by_split.to_csv(summary_by_split_path, index=False, encoding="utf-8-sig")

    summary_overall = (
        report_df
        .groupby(["triage_decision"])
        .agg(
            count=("triage_decision", "size"),
            mean_risk_score=("risk_score", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_entropy=("entropy", "mean"),
            mean_quality_score=("quality_score", "mean"),
            mean_lesion_burden=("total_lesion_burden", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    summary_overall_path = output_table_dir / "safety_triage_summary_overall.csv"
    summary_overall.to_csv(summary_overall_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.bar(summary_overall["triage_decision"], summary_overall["count"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Number of images")
    plt.title("RetinaGuard Safety Gate - Triage Decisions")
    plt.tight_layout()
    decision_fig_path = output_figure_dir / "triage_decision_counts.png"
    plt.savefig(decision_fig_path, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(report_df["risk_score"], bins=30)
    plt.xlabel("Risk score")
    plt.ylabel("Number of images")
    plt.title("RetinaGuard Safety Gate - Risk Score Distribution")
    plt.tight_layout()
    risk_fig_path = output_figure_dir / "risk_score_distribution.png"
    plt.savefig(risk_fig_path, dpi=200)
    plt.close()

    panel_df = report_df[
        (report_df["dataset"] == args.panel_dataset) &
        (report_df["split"] == args.panel_split)
    ].copy()

    if len(panel_df) > 0:
        panel_parts = []

        for decision in [
            "safe_negative_prediction",
            "low_risk_follow_up",
            "follow_up_recommended",
            "routine_referral",
            "urgent_referral",
            "manual_review_required",
        ]:
            part = panel_df[panel_df["triage_decision"] == decision].copy()
            if len(part) > 0:
                panel_parts.append(part.sample(n=min(3, len(part)), random_state=args.seed))

        if panel_parts:
            selected_panels = pd.concat(panel_parts, ignore_index=True)
        else:
            selected_panels = panel_df.sample(
                n=min(args.num_panels, len(panel_df)),
                random_state=args.seed,
            )

        if len(selected_panels) > args.num_panels:
            selected_panels = selected_panels.sample(
                n=args.num_panels,
                random_state=args.seed,
            ).reset_index(drop=True)

        panel_rows = []

        for idx, (_, row) in enumerate(selected_panels.iterrows()):
            safe_id = str(row["image_id"]).replace("/", "_").replace("\\", "_").replace(" ", "_")
            panel_path = panel_dir / f"{idx:02d}_{safe_id}_safety_triage.png"

            ok = make_case_panel(row, panel_path)

            if ok:
                panel_rows.append({
                    "image_id": row["image_id"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "triage_decision": row["triage_decision"],
                    "risk_score": row["risk_score"],
                    "panel_path": str(panel_path.relative_to(ROOT)),
                })

        panel_index_path = output_table_dir / "sample_triage_panels_index.csv"
        pd.DataFrame(panel_rows).to_csv(panel_index_path, index=False, encoding="utf-8-sig")

    readme_path = output_table_dir / "README_safety_gate.md"
    readme_path.write_text(
        "# RetinaGuard Safety Gate\n\n"
        "This stage adds an uncertainty-aware safety triage layer on top of the diabetic retinopathy pipeline.\n\n"
        "The safety gate combines:\n\n"
        "1. EfficientNet prediction confidence.\n"
        "2. Normalized prediction entropy.\n"
        "3. Top-2 class probability margin.\n"
        "4. Fundus image quality metrics.\n"
        "5. U-Net predicted lesion burden.\n"
        "6. Lesion-grade consistency.\n\n"
        "The output is not a clinical diagnosis. It is a research-oriented safety layer that decides whether a model prediction looks reliable, needs follow-up, requires routine referral, urgent referral, or manual review.\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("Safety Gate completed.")
    print("=" * 100)
    print("Saved:")
    print(report_path)
    print(summary_by_split_path)
    print(summary_overall_path)
    print(decision_fig_path)
    print(risk_fig_path)
    print(panel_dir)
    print(readme_path)
    print()
    print("Overall triage summary:")
    print(summary_overall.to_string(index=False))
    print()
    print("Split-level triage summary:")
    print(summary_by_split.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
