from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

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
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "message": "Prediction failed.",
            },
        )

    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
