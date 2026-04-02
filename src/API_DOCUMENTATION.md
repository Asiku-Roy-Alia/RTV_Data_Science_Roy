# RTV Field Image Classifier — API Documentation

## Overview

A production-ready RESTful API that classifies RTV field check-in images into
one of 9 programme categories using a fine-tuned EfficientNet-B0 model.

**Base URL (local):** `http://localhost:8000`  
**Interactive docs:** `http://localhost:8000/docs` (Swagger UI, auto-generated)

---

## Running the API

### Option 1 — Direct (development)

```bash
# From the project root
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000

# Override checkpoint location if needed
CHECKPOINT_PATH=../checkpoints/best_model.pth uvicorn api:app --port 8000
```

### Option 2 — Docker (recommended for deployment)

```bash
# Build
docker build -f Dockerfile -t rtv-classifier .

# Run — mount checkpoint from host
docker run -p 8000:8000 \
  -v $(pwd)/task2/checkpoints:/app/checkpoints:ro \
  rtv-classifier

# Or pass checkpoint path as env var
docker run -p 8000:8000 \
  -e CHECKPOINT_PATH=/app/checkpoints/best_model.pth \
  -v $(pwd)/task2/checkpoints:/app/checkpoints:ro \
  rtv-classifier
```

---

## Endpoints

### `POST /predict`

Classify a single field check-in image.

**Request**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file`    | `UploadFile` | Yes | JPEG, PNG, or WebP image |

**Accepted formats:** `image/jpeg`, `image/png`, `image/webp`  
**Maximum file size:** 10 MB

**Successful response — 200 OK**

```json
{
  "category":   "poultry-house",
  "confidence": 0.84,
  "status":     "success"
}
```

| Field        | Type    | Description |
|--------------|---------|-------------|
| `category`   | string  | Predicted programme category (one of 9 classes) |
| `confidence` | float   | Softmax probability of the predicted class (0–1) |
| `status`     | string  | Always `"success"` on 200 |

**Possible categories:**

```
compost  |  goat-sheep-pen  |  guinea-pig-shelter  |  liquid-organic
organic  |  pigsty          |  poultry-house       |  tippytap  |  vsla
```

**Error responses**

| Status | Condition | Response body |
|--------|-----------|---------------|
| `400`  | Empty file, corrupt image, or undecodable bytes | `{"status": "error", "message": "..."}` |
| `413`  | File exceeds 10 MB | `{"status": "error", "message": "File too large..."}` |
| `415`  | Unsupported file type (e.g. PDF, GIF) | `{"status": "error", "message": "Unsupported file type..."}` |
| `422`  | Missing `file` field in form data | FastAPI validation error |
| `500`  | Unexpected server error | `{"status": "error", "message": "An internal server error..."}` |
| `503`  | Model not yet loaded (startup in progress) | `{"status": "error", "message": "Model is not loaded..."}` |

---

### `GET /health`

Liveness check. Returns 200 when the server is running and the model is loaded.

**Response — 200 OK**
```json
{"status": "ok", "model_loaded": true}
```

**Response — 503 (model not loaded)**
```json
{"status": "unavailable", "detail": "Model not loaded"}
```

---

### `GET /model/info`

Metadata about the currently loaded model.

**Response — 200 OK**
```json
{
  "architecture":   "EfficientNet-B0",
  "num_classes":    9,
  "classes":        ["compost", "goat-sheep-pen", ...],
  "checkpoint":     "checkpoints/best_model.pth",
  "device":         "cpu",
  "input_size":     "224x224",
  "max_file_bytes": 10485760
}
```

---

## Example Requests

### curl

```bash
# Predict
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/field_photo.jpg" \
  | python -m json.tool

# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info
```

### Python (requests)

```python
import requests

with open("field_photo.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("field_photo.jpg", f, "image/jpeg")},
    )

result = response.json()
print(f"Category:   {result['category']}")
print(f"Confidence: {result['confidence']:.2%}")
# → Category:   tippytap
# → Confidence: 91.23%
```

### Python (httpx — async)

```python
import httpx, asyncio

async def classify(image_path: str) -> dict:
    async with httpx.AsyncClient() as client:
        with open(image_path, "rb") as f:
            r = await client.post(
                "http://localhost:8000/predict",
                files={"file": (image_path, f, "image/jpeg")},
            )
        r.raise_for_status()
        return r.json()

result = asyncio.run(classify("compost_photo.jpg"))
```

---

## Running Tests

```bash
pip install pytest httpx
pytest test_api.py -v

```

The test suite covers:
- All happy-path cases (JPEG, PNG, large images, small images, grayscale, RGBA)
- All defined error codes (400, 413, 415, 422)
- Response schema completeness
- Confidence range validation
- Model uses a mock — no checkpoint required to run tests

---

## Design Notes

**Why FastAPI?**  
FastAPI provides automatic OpenAPI/Swagger documentation, request validation via
Pydantic, async request handling, and type-safe response schemas — all in less
code than Flask with equivalent functionality.

**Why `workers=1` in the Dockerfile?**  
A single PyTorch model loaded into CPU/GPU memory is not thread-safe for
concurrent writes. With one worker and FastAPI's async I/O, the server handles
multiple concurrent requests safely — the model is loaded once and called
sequentially within the single process. For horizontal scaling under high load,
run multiple containers behind a load balancer.

**EXIF orientation**  
`ImageOps.exif_transpose()` is applied before inference, matching the training
pipeline. Field photos taken in portrait orientation and stored with an EXIF
rotation tag are correctly oriented before reaching the model.

**Confidence interpretation**  
The confidence score is the raw softmax probability of the top class. It is not
calibrated. A confidence of 0.60 does not mean "60% chance the prediction is
correct" — it means the model assigned 60% of its probability mass to that
class. For production use, temperature scaling calibration is recommended.
