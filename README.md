# RTV Field Image Classifier

**Candidate:** Roy Alia Asiku  
**Assessment:** Data Scientist Technical Assessment | Raising The Village  
**Date:** 2 April 2026  
**Repository:** [github.com/AsikuRoy-Alia/RTV_Data_Science_Roy](https://github.com/AsikuRoy-Alia/RTV_Data_Science_Roy)

---

## Overview

An end-to-end machine learning system that automatically classifies RTV field
check-in images into 9 programme categories:

`compost` · `goat-sheep-pen` · `guinea-pig-shelter` · `liquid-organic` · `organic` · `pigsty` · `poultry-house` · `tippytap` · `vsla`

**Architecture:** Fine-tuned EfficientNet-B0 · **Test weighted-F1:** 0.6854 · **Test accuracy:** 0.6721

---

## Project Structure

```
RTV_Data_Science_Roy/
│
├── requirements.txt          ← install everything from here
│
├── src/
│   ├── dataset.py            ← Task 1: data loading, augmentation, splits
│   ├── model.py              ← Task 2: EfficientNet-B0 architecture
│   ├── train.py              ← Task 2: two-phase training loop
│   ├── evaluate.py           ← Task 2: metrics, confusion matrix, report
│   ├── api.py                ← Task 3: FastAPI serving layer
│   ├── Dockerfile            ← Task 3/4: containerised API
│   ├── test_api.py           ← Task 3: 18 API tests (all passing)
│   │
│   ├── checkpoints/          ← created by train.py
│   │   ├── best_model.pth
│   │   ├── training_history.json
│   │   ├── evaluation_report.json
│   │   ├── confusion_matrix.png
│   │   └── confusion_matrix_raw.png
│   │
│   └── docs/
│       ├── task1_analysis_writeup.docx
│       └── task2_model_report.docx
│
└── ../data/                  ← dataset (outside repo, sibling directory)
    ├── compost/
    ├── goat-sheep-pen/
    ├── guinea-pig-shelter/
    ├── liquid-organic/
    ├── organic/
    ├── pigsty/
    ├── poultry-house/
    ├── tippytap/
    └── vsla/
```

---

## Quick Start

### 1. Install dependencies

```bash
# From the repo root (RTV_Data_Science_Roy/)

# CPU-only 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# GPU (CUDA 12.1) — faster training
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> **Windows note:** if `uvicorn` fails to start with an event-loop error,
> install the optional standard extras explicitly:
> `pip install "uvicorn[standard]"` and run with `--loop asyncio`:
> `uvicorn api:app --loop asyncio --host 0.0.0.0 --port 8000`

### 2. Task 1 — Data analysis

```bash
cd src
python dataset.py ../../data
# Outputs: analysis_report.json  
# Logs:    class distribution, imbalance ratio, PII flags, split sizes
```

### 3. Task 2 — Train

```bash
cd src
python train.py --data_dir ../../data

# Key options (all have defaults — bare invocation works):
#   --batch_size 32
#   --freeze_epochs 5      (head-only phase)
#   --max_epochs 30        (full budget; we ran 15 on laptop)
#   --checkpoint_dir checkpoints

# Outputs: checkpoints/best_model.pth
#          checkpoints/training_history.json
```

### 4. Task 2 — Evaluate

```bash
cd src
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir   ../../data \
    --output_dir checkpoints

# Outputs: checkpoints/evaluation_report.json
#          checkpoints/confusion_matrix.png       (row-normalised)
#          checkpoints/confusion_matrix_raw.png   (raw counts)
```

### 5. Task 3 — Run the API

```bash
cd src

# Standard
uvicorn api:app --host 0.0.0.0 --port 8000

# Windows (if event-loop error)
uvicorn api:app --loop asyncio --host 0.0.0.0 --port 8000

# Test it
curl -X POST http://localhost:8000/predict \
     -F "file=@/path/to/field_photo.jpg"

# → {"category": "tippytap", "confidence": 0.91, "status": "success"}
```

Interactive Swagger docs: **http://localhost:8000/docs**

### 6. Task 3 — Run tests

```bash
cd src
pytest test_api.py -v
# → 18 passed  (no checkpoint needed — model is mocked)
```

### 7. Task 3/4 — Docker

> **Windows prerequisite:** Docker Desktop must be running and the Linux engine
> must be active. If you see `permission denied while trying to connect to the
> Docker API`, open Docker Desktop, go to Settings → General, and ensure
> "Use the WSL 2 based engine" is enabled and the engine is running.
> Then re-open PowerShell and retry.

```bash
# From the repo root (RTV_Data_Science_Roy/)

# Build — run from repo root so COPY paths resolve correctly
docker build -f src/Dockerfile -t rtv-classifier .

# Run — mount checkpoint from host
docker run -p 8000:8000 \
  -v "$(pwd)/src/checkpoints:/app/checkpoints:ro" \
  rtv-classifier

# Windows PowerShell equivalent:
docker run -p 8000:8000 `
  -v "${PWD}\src\checkpoints:/app/checkpoints:ro" `
  rtv-classifier

# Override checkpoint path
docker run -p 8000:8000 \
  -e CHECKPOINT_PATH=/app/checkpoints/best_model.pth \
  -v "$(pwd)/src/checkpoints:/app/checkpoints:ro" \
  rtv-classifier
```

---

## Results Summary

| Metric              | Value  |
|---------------------|--------|
| Test accuracy       | 0.6721 |
| Test weighted F1    | **0.6854** |
| Test macro F1       | 0.6156 |
| Best epoch          | 11 / 15 |
| Training epochs run | 15 (of planned 30) |

**Per-class highlights:**

| Class              | F1     | Notes |
|--------------------|--------|-------|
| vsla               | 0.9333 | Visually distinct indoor setting |
| tippytap           | 0.9091 | Unique structural shape |
| compost            | 0.9091 | Distinctive texture |
| pigsty             | 0.6341 | Some confusion with goat-sheep-pen |
| poultry-house      | 0.6154 | Some confusion with goat-sheep-pen |
| liquid-organic     | 0.5306 | Near-mirror confusion with organic |
| organic            | 0.5306 | Near-mirror confusion with liquid-organic |
| goat-sheep-pen     | 0.4783 | Errors spread across enclosure classes |
| guinea-pig-shelter | 0.0000 | 2 test samples — statistically unresolvable |

Full analysis in `src/docs/task2_model_report.docx`.

---

## API Reference

| Endpoint       | Method | Description                        |
|----------------|--------|------------------------------------|
| `/predict`     | POST   | Classify an image (JPEG/PNG)       |
| `/health`      | GET    | Liveness check                     |
| `/model/info`  | GET    | Model metadata and class list      |
| `/docs`        | GET    | Interactive Swagger UI             |

**Request:**
```
POST /predict
Content-Type: multipart/form-data
Body: file=<image.jpg>
```

**Response:**
```json
{"category": "poultry-house", "confidence": 0.84, "status": "success"}
```

**Error codes:** `400` invalid/corrupt image · `413` file too large (>10 MB) ·
`415` unsupported type · `503` model not loaded

---

## Design Decisions

### Why EfficientNet-B0?
Best accuracy-to-parameter ratio for this scale (5.3M params, 82.0% ImageNet
top-1). ResNet-50 has 5× more parameters and lower accuracy. The compound
scaling design handles multi-scale features important for distinguishing
structurally similar animal enclosures.

### Why two-phase fine-tuning?
Phase 1 (frozen backbone, 5 epochs) warms up the randomly initialised head
without destroying pretrained features. Phase 2 (full unfreeze) then fine-tunes
with discriminative learning rates (backbone 10× lower than head). The training
history confirms this: val F1 jumped from 0.367→0.523 at the phase transition
(epoch 5→6), the largest single-epoch improvement in the run.

### Why WeightedRandomSampler + weighted loss?
The 9.38× class imbalance (guinea-pig-shelter=16, others=150) requires
intervention at both data and loss levels. The sampler rebalances the training
distribution per epoch; the weighted CrossEntropyLoss ensures gradient
contributions are proportional to class rarity. Both operate simultaneously.

### Why weighted F1 as the primary metric?
Accuracy is misleading under 9.38× imbalance — a model ignoring
guinea-pig-shelter entirely still achieves >97% accuracy. Weighted F1 accounts
for class frequency while remaining interpretable.

### Why `workers=1` in the API?
A single PyTorch model instance is not safe for concurrent state mutation.
One Uvicorn worker with async I/O handles concurrent requests safely without
sharing model state. Horizontal scaling at higher load is via multiple
containers behind a load balancer, not multiple workers per process.

---

## Time Spent

| Task | Time |
|------|------|
| Task 1: Data analysis & pipeline | ~0.5 hours |
| Task 2: Model training & evaluation | ~1.5 hours  |
| Task 3: API implementation & tests | ~1 hour |
| Task 4: Dockerfile & documentation | ~0.5 hours |
| **Total** | **~3.5 hours** |

---

## Assumptions & Notable Decisions

1. **Dataset path is a sibling directory** (`../data` relative to `src/`). The
   `--data_dir` flag on `train.py` and `evaluate.py` allows overriding this.

2. **vsla images have no file extension.** `dataset.py` explicitly handles
   extensionless files. A naive `glob("*.jpg")` would silently drop all 150
   vsla images.

3. **PII in vsla class.** Several vsla images contain visible client names and
   financial records. These are flagged in the scan log but not removed — data
   governance is RTV's decision, not a modelling decision. See Task 1 writeup
   §1.4.

4. **guinea-pig-shelter data quality.** The 16-image class contains at least
   one completely black image, several cartoon images, and multiple mislabelled
   images (greenhouse, bowl of produce). These are documented in Task 1 §1.5.
   Removing mislabelled images is recommended before the next training run.

5. **EXIF orientation.** `ImageOps.exif_transpose()` is applied at load time
   in both training and inference. Without this, portrait photos stored with
   rotation metadata arrive at the model sideways.

6. **Label smoothing (0.1).** Reduces overconfident predictions and is
   especially helpful for the guinea-pig-shelter class where the model has very
   few examples to learn from.

7. **Docker on Windows.** Docker requires Docker Desktop with the WSL 2 engine
   running. The `permission denied` error indicates the engine is not active,
   not a code error. See the Quick Start section above.

---

## Further Improvements

- **Run to 30 epochs**: Model not yet converged at epoch 15; early stopping
  had not triggered. Expected val F1 gain: ~0.03–0.05.
- **Merge or relabel organic / liquid-organic**: 19 of the model's errors
  are the near-symmetric organic↔liquid-organic confusion. These classes are
  visually almost indistinguishable in field photos.
- **Collect more guinea-pig-shelter images**: 16 images is below the
  practical minimum for supervised learning. 50 clean images would allow
  meaningful evaluation.
- **Clean guinea-pig-shelter labels**: Remove the black image, cartoons, and
  mislabelled subjects before the next training run.
- **Test-time augmentation (TTA)**: Average predictions over 5–10 augmented
  views at inference; free accuracy improvement with no retraining.
- **MixUp / CutMix**: Interpolation-based augmentation; particularly
  effective for the organic/liquid-organic boundary.
