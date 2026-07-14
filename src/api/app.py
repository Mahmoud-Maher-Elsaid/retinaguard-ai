from pathlib import Path
import base64
import io
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

from src.inference.retinaguard_single_image import predict_image


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="RetinaGuard-AI API",
    description="Research API for diabetic retinopathy grading, lesion analysis, and safety triage.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def web_app():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "retinaguard-ai-api",
        "message": "RetinaGuard-AI API is running.",
    }


def build_demo_overlay(image_path: Path) -> tuple[str, int]:
    try:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((900, 650))

        draw = ImageDraw.Draw(image)
        width, height = image.size

        boxes = [
            ("HE", 0.18, 0.58, 0.32, 0.72, "red"),
            ("EX", 0.60, 0.30, 0.76, 0.45, "yellow"),
            ("MA", 0.42, 0.50, 0.50, 0.58, "cyan"),
            ("SE", 0.66, 0.62, 0.82, 0.78, "magenta"),
        ]

        for label, x1, y1, x2, y2, color in boxes:
            left = int(x1 * width)
            top = int(y1 * height)
            right = int(x2 * width)
            bottom = int(y2 * height)
            draw.rectangle([left, top, right, bottom], outline=color, width=4)
            draw.text((left + 4, max(0, top - 14)), label, fill=color)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}", len(boxes)

    except Exception:
        return "", 0


def build_demo_result(image_path: Path, original_error: str) -> dict:
    overlay_base64, box_count = build_demo_overlay(image_path)

    return {
        "demo_mode": True,
        "real_inference_available": False,
        "fallback_reason": "Real model inference failed, so the API returned a research demo result for interface testing.",
        "original_error": original_error,
        "device": "demo-fallback",
        "classification": {
            "predicted_label": 2,
            "predicted_label_name": "Moderate",
            "confidence": 0.873,
            "top2_margin": 0.512,
            "probabilities": {
                "0": {"label_name": "No DR", "probability": 0.041},
                "1": {"label_name": "Mild", "probability": 0.066},
                "2": {"label_name": "Moderate", "probability": 0.873},
                "3": {"label_name": "Severe", "probability": 0.016},
                "4": {"label_name": "Proliferative DR", "probability": 0.004},
            },
        },
        "image_quality": {
            "image_quality_status": "acceptable",
            "quality_score": 0.921,
        },
        "segmentation": {
            "total_lesion_union_area_ratio": 0.074,
            "lesion_box_count": box_count,
            "annotated_image_base64": overlay_base64,
            "lesions": {
                "microaneurysms": {
                    "area_ratio": 0.008,
                    "probability_max": 0.741,
                },
                "haemorrhages": {
                    "area_ratio": 0.026,
                    "probability_max": 0.813,
                },
                "hard_exudates": {
                    "area_ratio": 0.034,
                    "probability_max": 0.862,
                },
                "soft_exudates": {
                    "area_ratio": 0.006,
                    "probability_max": 0.584,
                },
            },
        },
        "safety_gate": {
            "triage_decision": "routine_referral",
            "risk_score": 0.681,
            "uncertainty_level": "low",
            "lesion_evidence_level": "moderate",
            "lesion_grade_consistency": "consistent",
        },
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = Path(temp_file.name)

    try:
        result = predict_image(
            image_path=temp_path,
            amp=True,
        )
        return JSONResponse(content=result)

    except Exception as exc:
        demo_result = build_demo_result(
            image_path=temp_path,
            original_error=str(exc),
        )
        return JSONResponse(content=demo_result)

    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
