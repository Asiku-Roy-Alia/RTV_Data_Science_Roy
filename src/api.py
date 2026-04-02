"""
api.py — RTV Field Image Classifier — Production API
======================================================
FastAPI service that accepts a field check-in image via HTTP POST and returns
the predicted category and confidence score.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

    # or via Docker (see Dockerfile)
    docker build -t rtv-classifier .
    docker run -p 8000:8000 rtv-classifier

Endpoints:
    POST /predict          — classify a single uploaded image
    GET  /health           — liveness check
    GET  /model/info       — metadata about the loaded model
    GET  /docs             — interactive Swagger UI (auto-generated)

Expected response (POST /predict):
    {
        "category":   "poultry-house",
        "confidence": 0.84,
        "status":     "success"
    }
"""

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────
import os

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoints/best_model.pth")
IMAGE_SIZE      = 224
MAX_FILE_BYTES  = 10 * 1024 * 1024       # 10 MB hard limit
ALLOWED_TYPES   = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_SUFFIXES= {".jpg", ".jpeg", ".png", ".webp", ""}   # "" for vsla-style

# ImageNet normalisation — must match training pipeline
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── preprocessing transform (identical to val/test in dataset.py) ─────────────
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.143)),    # 256
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── model container (loaded once at startup) ──────────────────────────────────

class _ModelState:
    """
    Holds the loaded model, class mapping, and device.
    Populated during lifespan startup; read-only during inference.
    Thread-safe for reads (inference only mutates no shared state).
    """
    model:        Optional[nn.Module] = None
    idx_to_class: Optional[dict]      = None
    class_names:  Optional[list]      = None
    device:       Optional[torch.device] = None
    checkpoint_path: str = CHECKPOINT_PATH
    loaded_at:    Optional[float]     = None


STATE = _ModelState()


def _build_model(num_classes: int, dropout: float = 0.3) -> nn.Module:
    """
    Reconstruct EfficientNet-B0 with the custom 9-class head.
    Called once at startup. Weights=None avoids a download — checkpoint
    provides all weights including the backbone.
    """
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model(checkpoint_path: str) -> None:
    """
    Load checkpoint into STATE. Called once during lifespan startup.
    Raises RuntimeError if the checkpoint is missing or malformed.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise RuntimeError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first to generate checkpoints/best_model.pth"
        )

    STATE.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading checkpoint: %s  (device=%s)", checkpoint_path, STATE.device)

    ckpt = torch.load(checkpoint_path, map_location=STATE.device)

    # Validate checkpoint structure
    for key in ("model_state", "idx_to_class", "val_f1"):
        if key not in ckpt:
            raise RuntimeError(f"Checkpoint missing expected key: '{key}'")

    STATE.idx_to_class = ckpt["idx_to_class"]
    STATE.class_names  = [STATE.idx_to_class[i] for i in range(len(STATE.idx_to_class))]
    cfg                = ckpt.get("cfg", {})

    STATE.model = _build_model(
        num_classes = len(STATE.class_names),
        dropout     = cfg.get("dropout", 0.3),
    )
    STATE.model.load_state_dict(ckpt["model_state"])
    STATE.model.to(STATE.device)
    STATE.model.eval()
    STATE.loaded_at = time.time()

    n_params = sum(p.numel() for p in STATE.model.parameters())
    log.info(
        "Model loaded — classes=%d  params=%s  val_F1=%.4f  epoch=%d",
        len(STATE.class_names),
        f"{n_params:,}",
        ckpt["val_f1"],
        ckpt.get("epoch", -1),
    )


# ── FastAPI lifespan (startup / shutdown) ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model weights once at startup; release on shutdown.
    Using the lifespan context manager (FastAPI 0.93+) is preferred over
    the deprecated @app.on_event("startup") pattern.
    """
    load_model(STATE.checkpoint_path)
    yield
    # Cleanup on shutdown
    STATE.model = None
    log.info("Model unloaded — server shutting down")


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "RTV Field Image Classifier",
    description = (
        "Classifies RTV field check-in images into one of 9 programme categories. "
        "Powered by a fine-tuned EfficientNet-B0 model."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── response / error schemas ──────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    category:   str
    confidence: float
    status:     str = "success"

    model_config = {"json_schema_extra": {
        "example": {"category": "poultry-house", "confidence": 0.84, "status": "success"}
    }}


class ErrorResponse(BaseModel):
    status:  str = "error"
    message: str

    model_config = {"json_schema_extra": {
        "example": {"status": "error", "message": "Unsupported file type. Upload a JPEG or PNG image."}
    }}


# ── input validation helpers ──────────────────────────────────────────────────

def _validate_upload(file: UploadFile) -> None:
    """
    Raise HTTPException with appropriate status code for invalid inputs.

    Status codes:
        400 — missing filename or content
        413 — file exceeds size limit
        415 — unsupported media type
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    suffix = Path(file.filename).suffix.lower()
    ct     = (file.content_type or "").lower()

    # Accept by content-type OR by file extension (for extensionless edge cases)
    type_ok   = ct in ALLOWED_TYPES
    suffix_ok = suffix in ALLOWED_SUFFIXES

    if not (type_ok or suffix_ok):
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ct or suffix}'. "
                "Upload a JPEG, PNG, or WebP image."
            ),
        )


def _open_image(raw_bytes: bytes, filename: str) -> Image.Image:
    """
    Decode image bytes → RGB PIL Image.
    Applies EXIF orientation correction (same as dataset.py _safe_open).
    Raises HTTPException(400) for corrupt or unreadable image data.
    """
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.verify()                            # catches truncated images
        img = Image.open(io.BytesIO(raw_bytes)) # re-open after verify
        img = ImageOps.exif_transpose(img)      # correct EXIF rotation
        img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, OSError, Exception) as exc:
        log.warning("Failed to decode image '%s': %s", filename, exc)
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image. File may be corrupt or not a valid image."
        )


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _predict(img: Image.Image) -> tuple[str, float]:
    """
    Run inference on a PIL Image.
    Returns (predicted_category, confidence_score).
    """
    tensor = INFER_TRANSFORM(img).unsqueeze(0).to(STATE.device)   # (1, 3, 224, 224)
    logits = STATE.model(tensor)                                   # (1, 9)
    probs  = torch.softmax(logits, dim=1).squeeze(0)              # (9,)
    top_idx        = int(probs.argmax())
    confidence     = round(float(probs[top_idx]), 4)
    category       = STATE.class_names[top_idx]
    return category, confidence


# ── routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model        = PredictionResponse,
    responses             = {
        400: {"model": ErrorResponse, "description": "Invalid or corrupt image"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    summary               = "Classify a field check-in image",
    description           = (
        "Upload a JPEG or PNG field photo. Returns the predicted programme "
        "category and the model's confidence score (0–1)."
    ),
)
async def predict(file: UploadFile = File(..., description="Field check-in image (JPEG / PNG)")):
    # ── guard: model must be loaded ───────────────────────────────────────
    if STATE.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Server may be starting up."
        )

    # ── validate file type ────────────────────────────────────────────────
    _validate_upload(file)

    # ── read and size-check ────────────────────────────────────────────────
    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw_bytes)//1024} KB). Maximum is {MAX_FILE_BYTES//1024} KB."
        )

    # ── decode image ──────────────────────────────────────────────────────
    img = _open_image(raw_bytes, file.filename)

    # ── inference ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    category, confidence = _predict(img)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    log.info(
        "Prediction: file='%s'  category='%s'  confidence=%.4f  latency=%.1fms",
        file.filename, category, confidence, elapsed_ms,
    )

    return PredictionResponse(category=category, confidence=confidence)


@app.get(
    "/health",
    summary     = "Liveness check",
    description = "Returns 200 if the server is running and the model is loaded.",
)
async def health():
    if STATE.model is None:
        return JSONResponse(
            status_code = 503,
            content     = {"status": "unavailable", "detail": "Model not loaded"},
        )
    return {"status": "ok", "model_loaded": True}


@app.get(
    "/model/info",
    summary     = "Model metadata",
    description = "Returns metadata about the currently loaded model.",
)
async def model_info():
    if STATE.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "architecture"  : "EfficientNet-B0",
        "num_classes"   : len(STATE.class_names),
        "classes"       : STATE.class_names,
        "checkpoint"    : STATE.checkpoint_path,
        "device"        : str(STATE.device),
        "input_size"    : f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "max_file_bytes": MAX_FILE_BYTES,
    }


# ── global exception handler ──────────────────────────────────────────────────

from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException as FastAPIHTTPException

@app.exception_handler(FastAPIHTTPException)
async def http_exception_handler(request: Request, exc: FastAPIHTTPException):
    """
    Normalise all HTTPException responses to {"status": "error", "message": ...}
    so the API has a consistent error shape across all endpoints and status codes.
    """
    return JSONResponse(
        status_code = exc.status_code,
        content     = {"status": "error", "message": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch any unhandled exception and return a structured JSON error response.
    Prevents raw Python tracebacks from leaking to the client.
    """
    log.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code = 500,
        content     = {
            "status" : "error",
            "message": "An internal server error occurred. Please try again.",
        },
    )


# ── dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
