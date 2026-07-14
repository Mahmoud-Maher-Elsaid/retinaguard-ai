from pathlib import Path
import argparse
import base64
import json
import sys

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.transforms import get_classification_transforms, get_segmentation_transforms
from src.models.classification_models import create_classifier
from src.models.segmentation_models import create_unet


DR_LABEL_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

LESION_NAMES = [
    "microaneurysms",
    "haemorrhages",
    "hard_exudates",
    "soft_exudates",
]

DEFAULT_THRESHOLDS = {
    "microaneurysms": 0.15,
    "haemorrhages": 0.40,
    "hard_exudates": 0.60,
    "soft_exudates": 0.75,
}


LESION_COLORS_RGB = {
    "microaneurysms": (255, 64, 64),
    "haemorrhages": (255, 135, 40),
    "hard_exudates": (255, 220, 60),
    "soft_exudates": (60, 220, 255),
}


LESION_DISPLAY_NAMES = {
    "microaneurysms": "Microaneurysms",
    "haemorrhages": "Haemorrhages",
    "hard_exudates": "Hard Exudates",
    "soft_exudates": "Soft Exudates",
}


def extract_lesion_boxes_from_masks(
    probabilities,
    thresholds,
    original_shape,
    min_area_pixels=18,
    max_boxes_per_lesion=12,
):
    original_h, original_w = original_shape[:2]
    model_h, model_w = probabilities.shape[1], probabilities.shape[2]

    all_boxes = []

    for lesion_index, lesion_name in enumerate(LESION_NAMES):
        threshold = thresholds[lesion_name]
        probability_map = probabilities[lesion_index]
        binary_mask = (probability_map >= threshold).astype("uint8")

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        lesion_boxes = []

        for contour in contours:
            area_pixels = float(cv2.contourArea(contour))

            if area_pixels < min_area_pixels:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            x1 = int(round((x / model_w) * original_w))
            y1 = int(round((y / model_h) * original_h))
            x2 = int(round(((x + w) / model_w) * original_w))
            y2 = int(round(((y + h) / model_h) * original_h))

            x1 = max(0, min(x1, original_w - 1))
            y1 = max(0, min(y1, original_h - 1))
            x2 = max(0, min(x2, original_w - 1))
            y2 = max(0, min(y2, original_h - 1))

            crop_prob = probability_map[y:y + h, x:x + w]

            lesion_boxes.append({
                "lesion_type": lesion_name,
                "lesion_name": LESION_DISPLAY_NAMES[lesion_name],
                "threshold": float(threshold),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "width": int(max(0, x2 - x1)),
                "height": int(max(0, y2 - y1)),
                "area_pixels_model": area_pixels,
                "area_ratio_model": float(area_pixels / float(model_h * model_w)),
                "mean_probability": float(crop_prob.mean()) if crop_prob.size else 0.0,
                "max_probability": float(crop_prob.max()) if crop_prob.size else 0.0,
            })

        lesion_boxes = sorted(
            lesion_boxes,
            key=lambda item: item["area_pixels_model"],
            reverse=True,
        )[:max_boxes_per_lesion]

        all_boxes.extend(lesion_boxes)

    all_boxes = sorted(
        all_boxes,
        key=lambda item: item["area_pixels_model"],
        reverse=True,
    )

    return all_boxes


def draw_lesion_boxes_on_image(image_rgb, lesion_boxes):
    canvas = image_rgb.copy()

    for box in lesion_boxes:
        lesion_type = box["lesion_type"]
        color = LESION_COLORS_RGB.get(lesion_type, (255, 255, 255))

        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])

        label = f'{box["lesion_name"]} {box["max_probability"]:.2f}'

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)

        label_y = max(22, y1 - 8)
        (label_w, label_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )

        cv2.rectangle(
            canvas,
            (x1, label_y - label_h - 8),
            (x1 + label_w + 10, label_y + 4),
            color,
            -1,
        )

        cv2.putText(
            canvas,
            label,
            (x1 + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (5, 5, 5),
            2,
            cv2.LINE_AA,
        )

    return canvas


def encode_image_to_base64_jpeg(image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )

    if not success:
        return None

    encoded = base64.b64encode(buffer).decode("utf-8")
    return "data:image/jpeg;base64," + encoded



def parse_args():
    parser = argparse.ArgumentParser(description="Run RetinaGuard-AI on one fundus image.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, default="checkpoints/efficientnet_b0_aptos_100ep_bs16_no_sampler/best_model.pt")
    parser.add_argument("--segmentation-checkpoint", type=str, default="checkpoints/unet_resnet34_idrid_100ep/best_model.pt")
    parser.add_argument("--thresholds-csv", type=str, default="reports/tables/unet_resnet34_idrid_100ep/best_thresholds_validation.csv")
    parser.add_argument("--classification-image-size", type=int, default=384)
    parser.add_argument("--segmentation-image-size", type=int, default=512)
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def resolve_path(path_value):
    path = Path(str(path_value))
    if path.exists():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    return path


def read_image_rgb(image_path):
    image_path = resolve_path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, image_path


def normalized_entropy(probabilities):
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(len(probs)))


def top2_margin(probabilities):
    probs = np.sort(np.asarray(probabilities, dtype=np.float64))[::-1]
    return float(probs[0] - probs[1])


def compute_image_quality(image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_bgr = cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    blur_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    blur_score = float(np.clip(blur_raw / 350.0, 0.0, 1.0))
    brightness_score = float(np.clip(1.0 - abs(brightness_mean - 115.0) / 115.0, 0.0, 1.0))
    contrast_score = float(np.clip(contrast_std / 65.0, 0.0, 1.0))

    quality_score = float(0.45 * blur_score + 0.30 * brightness_score + 0.25 * contrast_score)

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


def load_classifier(checkpoint_path, device):
    checkpoint_path = resolve_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = create_classifier(model_name=model_name, num_classes=5, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_name


def load_segmentation_model(checkpoint_path, encoder_name, device):
    checkpoint_path = resolve_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    attempts = [
        lambda: create_unet(encoder_name=encoder_name, classes=4, encoder_weights=None),
        lambda: create_unet(encoder_name=encoder_name, in_channels=3, classes=4, encoder_weights=None),
        lambda: create_unet(encoder_name=encoder_name, in_channels=3, classes=4, pretrained=False),
        lambda: create_unet(encoder_name=encoder_name),
    ]

    model = None
    last_error = None

    for attempt in attempts:
        try:
            model = attempt()
            break
        except Exception as exc:
            last_error = exc

    if model is None:
        raise RuntimeError(f"Could not create U-Net model. Last error: {last_error}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_thresholds(thresholds_csv):
    thresholds = DEFAULT_THRESHOLDS.copy()
    path = resolve_path(thresholds_csv)

    if not path.exists():
        return thresholds

    df = pd.read_csv(path)
    lower_cols = {col.lower(): col for col in df.columns}

    lesion_col = None
    threshold_col = None

    for candidate in ["lesion", "class", "lesion_name", "channel"]:
        if candidate in lower_cols:
            lesion_col = lower_cols[candidate]
            break

    for candidate in ["best_threshold", "threshold", "selected_threshold"]:
        if candidate in lower_cols:
            threshold_col = lower_cols[candidate]
            break

    if lesion_col is None or threshold_col is None:
        return thresholds

    for _, row in df.iterrows():
        lesion_text = str(row[lesion_col]).lower()
        for lesion in LESION_NAMES:
            if lesion in lesion_text:
                thresholds[lesion] = float(row[threshold_col])

    return thresholds


@torch.no_grad()
def classify_image(model, image_rgb, image_size, device, amp_enabled):
    transform = get_classification_transforms(image_size=image_size, train=False)
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda"):
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_label = int(np.argmax(probs))

    return {
        "predicted_label": pred_label,
        "predicted_label_name": DR_LABEL_NAMES[pred_label],
        "confidence": float(np.max(probs)),
        "entropy": normalized_entropy(probs),
        "top2_margin": top2_margin(probs),
        "probabilities": {
            str(i): {
                "label_name": DR_LABEL_NAMES[i],
                "probability": float(probs[i]),
            }
            for i in range(5)
        },
    }


@torch.no_grad()
def segment_lesions(model, image_rgb, image_size, thresholds, device, amp_enabled):
    transform = get_segmentation_transforms(image_size=image_size, train=False)
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda"):
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

    lesion_summary = {}
    total_union = np.zeros_like(probs[0], dtype=bool)

    for idx, lesion in enumerate(LESION_NAMES):
        threshold = thresholds[lesion]
        mask = probs[idx] >= threshold
        total_union = np.logical_or(total_union, mask)

        lesion_summary[lesion] = {
            "threshold": float(threshold),
            "area_ratio": float(mask.mean()),
            "probability_mean": float(probs[idx].mean()),
            "probability_max": float(probs[idx].max()),
        }

    lesion_boxes = extract_lesion_boxes_from_masks(
        probabilities=probs,
        thresholds=thresholds,
        original_shape=image_rgb.shape,
        min_area_pixels=18,
        max_boxes_per_lesion=12,
    )

    annotated_image = draw_lesion_boxes_on_image(
        image_rgb=image_rgb,
        lesion_boxes=lesion_boxes,
    )

    annotated_image_base64 = encode_image_to_base64_jpeg(annotated_image)

    return {
        "lesions": lesion_summary,
        "total_lesion_union_area_ratio": float(total_union.mean()),
        "lesion_boxes": lesion_boxes,
        "lesion_box_count": int(len(lesion_boxes)),
        "annotated_image_base64": annotated_image_base64,
    }


def lesion_evidence_level(total_burden):
    if total_burden >= 0.025:
        return "very_high"
    if total_burden >= 0.012:
        return "high"
    if total_burden >= 0.004:
        return "medium"
    if total_burden >= 0.001:
        return "low"
    return "very_low"


def lesion_grade_consistency(pred_label, lesion_level):
    if pred_label == 0:
        return "consistent" if lesion_level in ["very_low", "low"] else "borderline" if lesion_level == "medium" else "inconsistent"

    if pred_label == 1:
        return "consistent" if lesion_level in ["low", "medium"] else "borderline" if lesion_level in ["very_low", "high"] else "inconsistent"

    if pred_label == 2:
        return "consistent" if lesion_level in ["medium", "high"] else "borderline" if lesion_level in ["low", "very_high"] else "inconsistent"

    if pred_label in [3, 4]:
        return "consistent" if lesion_level in ["high", "very_high"] else "borderline" if lesion_level == "medium" else "inconsistent"

    return "unknown"


def uncertainty_level(confidence, entropy, margin):
    if confidence >= 0.85 and entropy <= 0.35 and margin >= 0.45:
        return "low"
    if confidence >= 0.65 and entropy <= 0.60 and margin >= 0.20:
        return "medium"
    return "high"


def risk_score(pred_label, entropy, margin, quality_score, total_burden):
    severity_component = pred_label / 4.0
    uncertainty_component = entropy
    low_margin_component = 1.0 - np.clip(margin, 0.0, 1.0)
    quality_risk = 1.0 - quality_score
    lesion_component = np.clip(total_burden / 0.03, 0.0, 1.0)

    score = (
        0.35 * severity_component +
        0.20 * uncertainty_component +
        0.15 * low_margin_component +
        0.15 * quality_risk +
        0.15 * lesion_component
    )

    return float(np.clip(score, 0.0, 1.0))


def triage_decision(pred_label, confidence, uncertainty, image_quality_status, consistency, lesion_level, score):
    if image_quality_status == "poor":
        return "manual_review_required"

    if uncertainty == "high":
        return "manual_review_required"

    if consistency == "inconsistent":
        return "manual_review_required"

    if pred_label >= 3:
        return "urgent_referral"

    if pred_label == 2:
        if lesion_level in ["high", "very_high"] or score >= 0.55:
            return "urgent_referral"
        return "routine_referral"

    if pred_label == 1:
        return "follow_up_recommended"

    if pred_label == 0:
        if confidence >= 0.85 and lesion_level in ["very_low", "low"] and uncertainty == "low":
            return "safe_negative_prediction"
        return "low_risk_follow_up"

    return "manual_review_required"


def predict_image(
    image_path,
    classifier_checkpoint="checkpoints/efficientnet_b0_aptos_100ep_bs16_no_sampler/best_model.pt",
    segmentation_checkpoint="checkpoints/unet_resnet34_idrid_100ep/best_model.pt",
    thresholds_csv="reports/tables/unet_resnet34_idrid_100ep/best_thresholds_validation.csv",
    classification_image_size=384,
    segmentation_image_size=512,
    encoder_name="resnet34",
    amp=False,
    device=None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    image_rgb, resolved_image_path = read_image_rgb(image_path)

    classifier, classifier_name = load_classifier(classifier_checkpoint, device)
    segmentation_model = load_segmentation_model(segmentation_checkpoint, encoder_name, device)
    thresholds = load_thresholds(thresholds_csv)

    quality = compute_image_quality(image_rgb)

    classification = classify_image(
        model=classifier,
        image_rgb=image_rgb,
        image_size=classification_image_size,
        device=device,
        amp_enabled=amp,
    )

    segmentation = segment_lesions(
        model=segmentation_model,
        image_rgb=image_rgb,
        image_size=segmentation_image_size,
        thresholds=thresholds,
        device=device,
        amp_enabled=amp,
    )

    burden = segmentation["total_lesion_union_area_ratio"]
    level = lesion_evidence_level(burden)

    consistency = lesion_grade_consistency(
        pred_label=classification["predicted_label"],
        lesion_level=level,
    )

    uncertainty = uncertainty_level(
        confidence=classification["confidence"],
        entropy=classification["entropy"],
        margin=classification["top2_margin"],
    )

    score = risk_score(
        pred_label=classification["predicted_label"],
        entropy=classification["entropy"],
        margin=classification["top2_margin"],
        quality_score=quality["quality_score"],
        total_burden=burden,
    )

    decision = triage_decision(
        pred_label=classification["predicted_label"],
        confidence=classification["confidence"],
        uncertainty=uncertainty,
        image_quality_status=quality["image_quality_status"],
        consistency=consistency,
        lesion_level=level,
        score=score,
    )

    return {
        "project": "RetinaGuard-AI",
        "mode": "single_image_inference",
        "image_path": str(resolved_image_path),
        "device": str(device),
        "classifier_model": classifier_name,
        "classification": classification,
        "segmentation": segmentation,
        "image_quality": quality,
        "safety_gate": {
            "uncertainty_level": uncertainty,
            "lesion_evidence_level": level,
            "lesion_grade_consistency": consistency,
            "risk_score": score,
            "triage_decision": decision,
        },
        "disclaimer": "Research prototype only. Not for clinical diagnosis or medical decision-making.",
    }


def main():
    args = parse_args()

    result = predict_image(
        image_path=args.image,
        classifier_checkpoint=args.classifier_checkpoint,
        segmentation_checkpoint=args.segmentation_checkpoint,
        thresholds_csv=args.thresholds_csv,
        classification_image_size=args.classification_image_size,
        segmentation_image_size=args.segmentation_image_size,
        encoder_name=args.encoder_name,
        amp=args.amp,
    )

    result_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json, encoding="utf-8")
        print(f"Saved prediction JSON: {output_path}")

    print(result_json)


if __name__ == "__main__":
    main()
