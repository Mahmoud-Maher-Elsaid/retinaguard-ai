import argparse
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset_registry import load_aptos_split, load_idrid_grading_split
from src.data.transforms import get_classification_transforms, get_segmentation_transforms
from src.models.classification_models import create_classifier
from src.models.segmentation_models import create_unet


DR_LABELS = {
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--classifier-checkpoint",
        type=str,
        default="checkpoints/efficientnet_b0_aptos_100ep_bs16_no_sampler/best_model.pt",
    )
    parser.add_argument(
        "--segmentation-checkpoint",
        type=str,
        default="checkpoints/unet_resnet34_idrid_100ep/best_model.pt",
    )
    parser.add_argument(
        "--thresholds-csv",
        type=str,
        default="reports/tables/unet_resnet34_idrid_100ep/best_thresholds_validation.csv",
    )
    parser.add_argument("--dataset", type=str, default="APTOS2019", choices=["APTOS2019", "IDRiD"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--classification-image-size", type=int, default=384)
    parser.add_argument("--segmentation-image-size", type=int, default=512)
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="retinaguard_explainability_demo")

    return parser.parse_args()


def safe_path(path_value):
    path = Path(str(path_value))
    if path.exists():
        return path

    candidate = ROOT / path
    if candidate.exists():
        return candidate

    return path


def load_demo_dataframe(dataset_name, split, num_samples, seed):
    if dataset_name == "APTOS2019":
        df = load_aptos_split(split).copy()
        df["dataset"] = "APTOS2019"
    else:
        df = load_idrid_grading_split(split).copy()
        df["dataset"] = "IDRiD"

    df["split"] = split

    if "image_exists" in df.columns:
        df = df[df["image_exists"] == True].copy()

    df = df[df["label"].isin([0, 1, 2, 3, 4])].copy()
    df["label"] = df["label"].astype(int)

    rng = np.random.default_rng(seed)
    selected_parts = []

    per_class = max(1, int(np.ceil(num_samples / 5)))

    for label in [0, 1, 2, 3, 4]:
        part = df[df["label"] == label].copy()
        if len(part) == 0:
            continue

        n = min(per_class, len(part))
        chosen_idx = rng.choice(part.index.to_numpy(), size=n, replace=False)
        selected_parts.append(part.loc[chosen_idx])

    selected = pd.concat(selected_parts, ignore_index=True)

    if len(selected) > num_samples:
        selected = selected.sample(n=num_samples, random_state=seed).reset_index(drop=True)

    return selected.reset_index(drop=True)


def load_classifier(checkpoint_path, fallback_model_name, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", fallback_model_name)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = create_classifier(
        model_name=model_name,
        num_classes=5,
        pretrained=False,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, model_name


def load_segmentation_model(checkpoint_path, encoder_name, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = None

    attempts = [
        lambda: create_unet(encoder_name=encoder_name, classes=4, encoder_weights=None),
        lambda: create_unet(encoder_name=encoder_name, in_channels=3, classes=4, encoder_weights=None),
        lambda: create_unet(encoder_name=encoder_name, in_channels=3, classes=4, pretrained=False),
        lambda: create_unet(encoder_name=encoder_name),
    ]

    last_error = None
    for attempt in attempts:
        try:
            model = attempt()
            break
        except Exception as e:
            last_error = e

    if model is None:
        raise RuntimeError(f"Could not create U-Net model. Last error: {last_error}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def find_last_conv_layer(model):
    last_name = None
    last_module = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_module = module

    if last_module is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")

    return last_name, last_module


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        self.layer_name, self.target_layer = find_last_conv_layer(model)

        self.forward_handle = self.target_layer.register_forward_hook(self.forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, target_class=None):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(image_tensor)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[:, target_class].sum()
        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, logits.detach()

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def read_image_rgb(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def make_gradcam_overlay(image_rgb, cam):
    h, w = image_rgb.shape[:2]

    cam_resized = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.55 * image_rgb + 0.45 * heatmap).clip(0, 255).astype(np.uint8)
    return overlay


def load_thresholds(thresholds_csv):
    thresholds = DEFAULT_THRESHOLDS.copy()
    path = ROOT / thresholds_csv

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
def predict_lesions(model, image_rgb, transform, thresholds, device, amp_enabled):
    transformed = transform(image=image_rgb)
    tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda"):
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

    masks = {}

    for idx, lesion in enumerate(LESION_NAMES):
        threshold = thresholds[lesion]
        mask = probs[idx] >= threshold
        masks[lesion] = {
            "prob": probs[idx],
            "mask": mask.astype(np.uint8),
            "threshold": threshold,
            "area_ratio": float(mask.mean()),
            "prob_mean": float(probs[idx].mean()),
            "prob_max": float(probs[idx].max()),
        }

    return masks


def make_lesion_overlay(image_rgb, lesion_masks):
    h, w = image_rgb.shape[:2]
    overlay = image_rgb.copy().astype(np.float32)

    colors = {
        "microaneurysms": np.array([255, 0, 0], dtype=np.float32),
        "haemorrhages": np.array([255, 128, 0], dtype=np.float32),
        "hard_exudates": np.array([255, 255, 0], dtype=np.float32),
        "soft_exudates": np.array([0, 255, 255], dtype=np.float32),
    }

    alpha = 0.45

    for lesion, info in lesion_masks.items():
        mask = info["mask"]
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        color = colors[lesion]
        overlay[mask_resized] = (1 - alpha) * overlay[mask_resized] + alpha * color

    return overlay.clip(0, 255).astype(np.uint8)


def save_demo_panel(
    image_rgb,
    gradcam_overlay,
    lesion_overlay,
    probs,
    true_label,
    pred_label,
    confidence,
    lesion_masks,
    output_path,
):
    labels = [DR_LABELS[i] for i in range(5)]
    prob_values = probs.tolist()

    fig = plt.figure(figsize=(15, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(image_rgb)
    ax1.set_title(f"Original\nTrue: {true_label} - {DR_LABELS[true_label]}")
    ax1.axis("off")

    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(gradcam_overlay)
    ax2.set_title(f"EfficientNet Grad-CAM\nPred: {pred_label} - {DR_LABELS[pred_label]} | Conf: {confidence:.3f}")
    ax2.axis("off")

    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(lesion_overlay)
    ax3.set_title("U-Net Predicted Lesion Overlay")
    ax3.axis("off")

    ax4 = plt.subplot(2, 2, 4)
    y_positions = np.arange(len(labels))
    ax4.barh(y_positions, prob_values)
    ax4.set_yticks(y_positions)
    ax4.set_yticklabels(labels)
    ax4.set_xlim(0, 1)
    ax4.set_xlabel("Probability")
    ax4.set_title("DR Class Probabilities")

    lesion_text = []
    for lesion, info in lesion_masks.items():
        lesion_text.append(
            f"{lesion}: area={info['area_ratio']:.4f}, maxP={info['prob_max']:.3f}"
        )

    fig.text(
        0.02,
        0.02,
        "Lesion summary: " + " | ".join(lesion_text),
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = ROOT / "reports" / "figures" / args.run_name
    table_dir = ROOT / "reports" / "tables" / args.run_name

    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("RetinaGuard-AI Stage 8: Explainability Demo")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Output figures: {output_dir}")
    print(f"Output tables: {table_dir}")
    print("=" * 100)

    df = load_demo_dataframe(
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    classifier, classifier_name = load_classifier(
        checkpoint_path=ROOT / args.classifier_checkpoint,
        fallback_model_name="efficientnet_b0",
        device=device,
    )

    segmentation_model = load_segmentation_model(
        checkpoint_path=ROOT / args.segmentation_checkpoint,
        encoder_name=args.encoder_name,
        device=device,
    )

    thresholds = load_thresholds(args.thresholds_csv)

    classification_transform = get_classification_transforms(
        image_size=args.classification_image_size,
        train=False,
    )

    segmentation_transform = get_segmentation_transforms(
        image_size=args.segmentation_image_size,
        train=False,
    )

    gradcam = GradCAM(classifier)

    summary_rows = []

    for index, row in df.iterrows():
        image_path = safe_path(row["image_path"])
        image_rgb = read_image_rgb(image_path)

        transformed = classification_transform(image=image_rgb)
        image_tensor = transformed["image"].unsqueeze(0).to(device)

        cam, logits = gradcam.generate(image_tensor=image_tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_label = int(np.argmax(probs))
        confidence = float(np.max(probs))
        true_label = int(row["label"])

        gradcam_overlay = make_gradcam_overlay(image_rgb, cam)

        lesion_masks = predict_lesions(
            model=segmentation_model,
            image_rgb=image_rgb,
            transform=segmentation_transform,
            thresholds=thresholds,
            device=device,
            amp_enabled=args.amp,
        )

        lesion_overlay = make_lesion_overlay(image_rgb, lesion_masks)

        image_id = str(row["image_id"])
        safe_image_id = image_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
        figure_path = output_dir / f"{index:02d}_{safe_image_id}_explainability.png"

        save_demo_panel(
            image_rgb=image_rgb,
            gradcam_overlay=gradcam_overlay,
            lesion_overlay=lesion_overlay,
            probs=probs,
            true_label=true_label,
            pred_label=pred_label,
            confidence=confidence,
            lesion_masks=lesion_masks,
            output_path=figure_path,
        )

        summary = {
            "dataset": args.dataset,
            "split": args.split,
            "image_id": image_id,
            "image_path": str(image_path),
            "true_label": true_label,
            "true_label_name": DR_LABELS[true_label],
            "pred_label": pred_label,
            "pred_label_name": DR_LABELS[pred_label],
            "confidence": confidence,
            "correct": int(true_label == pred_label),
            "figure_path": str(figure_path.relative_to(ROOT)),
        }

        for class_id in range(5):
            summary[f"prob_{class_id}"] = float(probs[class_id])

        for lesion, info in lesion_masks.items():
            summary[f"{lesion}_area_ratio"] = info["area_ratio"]
            summary[f"{lesion}_prob_mean"] = info["prob_mean"]
            summary[f"{lesion}_prob_max"] = info["prob_max"]
            summary[f"{lesion}_threshold"] = info["threshold"]

        summary_rows.append(summary)

        print(
            f"[{index + 1}/{len(df)}] {image_id} | "
            f"true={true_label} {DR_LABELS[true_label]} | "
            f"pred={pred_label} {DR_LABELS[pred_label]} | "
            f"conf={confidence:.3f} | saved={figure_path.name}"
        )

    gradcam.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_path = table_dir / "explainability_demo_predictions.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    readme_path = table_dir / "README_explainability_demo.md"
    readme_path.write_text(
        "# RetinaGuard-AI Explainability Demo\n\n"
        "This folder contains sample visual explanations generated by the final RetinaGuard-AI pipeline.\n\n"
        "Each panel contains:\n\n"
        "1. Original fundus image.\n"
        "2. EfficientNet Grad-CAM heatmap.\n"
        "3. U-Net predicted lesion overlay.\n"
        "4. DR class probability distribution.\n\n"
        "This demo is intended for qualitative interpretation only, not clinical use.\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("Explainability demo completed.")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved figures to: {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
