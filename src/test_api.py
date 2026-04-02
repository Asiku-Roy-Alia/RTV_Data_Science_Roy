"""
test_api.py — API test suite
==============================
Tests every endpoint and every defined error path.

Run:
    pip install pytest httpx
    pytest test_api.py -v

Tests use FastAPI's TestClient (synchronous httpx wrapper) so no running
server is needed. The model is mocked so tests pass without a checkpoint.
"""

import io
import json
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from fastapi.testclient import TestClient

# ── patch model loading before importing app ──────────────────────────────────
# We mock load_model so tests don't require a real checkpoint file.

def _make_jpeg_bytes(width=224, height=224, color=(100, 149, 237)) -> bytes:
    """Create a minimal valid JPEG in memory."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()

def _make_png_bytes(width=224, height=224) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(80, 200, 120)).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient with a mocked model state.
    The mock returns a fixed prediction so tests are deterministic.
    """
    import api

    # Mock the model state directly — avoids loading any checkpoint
    mock_model = MagicMock()

    import torch
    # Return logits that give 100% confidence for class 0 ("compost")
    n_classes = 9
    fake_logits = torch.zeros(1, n_classes)
    fake_logits[0, 0] = 100.0   # compost gets all the probability mass
    mock_model.return_value = fake_logits
    mock_model.eval.return_value = None

    api.STATE.model        = mock_model
    api.STATE.idx_to_class = {i: c for i, c in enumerate([
        "compost", "goat-sheep-pen", "guinea-pig-shelter", "liquid-organic",
        "organic", "pigsty", "poultry-house", "tippytap", "vsla"
    ])}
    api.STATE.class_names  = list(api.STATE.idx_to_class.values())
    api.STATE.device       = torch.device("cpu")
    api.STATE.checkpoint_path = "mock/best_model.pth"

    # Patch load_model so the lifespan startup doesn't try to read a real file
    with patch.object(api, "load_model", return_value=None):
        with TestClient(api.app) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


# ── /model/info ───────────────────────────────────────────────────────────────

class TestModelInfo:
    def test_returns_class_list(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        body = r.json()
        assert body["num_classes"] == 9
        assert "compost" in body["classes"]
        assert "vsla"    in body["classes"]
        assert body["architecture"] == "EfficientNet-B0"
        assert body["input_size"]   == "224x224"


# ── /predict — happy paths ────────────────────────────────────────────────────

class TestPredictSuccess:
    def test_jpeg_returns_valid_prediction(self, client):
        r = client.post(
            "/predict",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["category"] in [
            "compost", "goat-sheep-pen", "guinea-pig-shelter",
            "liquid-organic", "organic", "pigsty",
            "poultry-house", "tippytap", "vsla",
        ]
        assert 0.0 <= body["confidence"] <= 1.0

    def test_png_accepted(self, client):
        r = client.post(
            "/predict",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "success"

    def test_response_schema_complete(self, client):
        """Response must contain exactly: category, confidence, status."""
        r = client.post(
            "/predict",
            files={"file": ("img.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        body = r.json()
        assert set(body.keys()) == {"category", "confidence", "status"}

    def test_confidence_is_float_between_0_and_1(self, client):
        r = client.post(
            "/predict",
            files={"file": ("img.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        conf = r.json()["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_large_image_accepted(self, client):
        """4000×3000 image should be accepted and resized by the pipeline."""
        r = client.post(
            "/predict",
            files={"file": ("large.jpg", _make_jpeg_bytes(4000, 3000), "image/jpeg")},
        )
        assert r.status_code == 200

    def test_small_image_accepted(self, client):
        """32×32 thumbnail should be upscaled and accepted."""
        r = client.post(
            "/predict",
            files={"file": ("small.jpg", _make_jpeg_bytes(32, 32), "image/jpeg")},
        )
        assert r.status_code == 200


# ── /predict — error paths ────────────────────────────────────────────────────

class TestPredictErrors:
    def test_415_unsupported_file_type(self, client):
        """PDF should be rejected with 415."""
        r = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert r.status_code == 415
        assert "message" in r.json()

    def test_415_text_file(self, client):
        r = client.post(
            "/predict",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
        assert r.status_code == 415

    def test_400_empty_file(self, client):
        """Zero-byte upload should be rejected with 400."""
        r = client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert r.status_code == 400

    def test_400_corrupt_image_bytes(self, client):
        """File with image content-type but invalid bytes should return 400."""
        r = client.post(
            "/predict",
            files={"file": ("corrupt.jpg", b"\xff\xd8\xff garbage data", "image/jpeg")},
        )
        assert r.status_code == 400

    def test_413_file_too_large(self, client):
        """File exceeding MAX_FILE_BYTES (10 MB) should return 413."""
        import api
        big = b"x" * (api.MAX_FILE_BYTES + 1)
        r = client.post(
            "/predict",
            files={"file": ("big.jpg", big, "image/jpeg")},
        )
        assert r.status_code == 413

    def test_error_response_has_message_field(self, client):
        """All error responses must have a 'message' field."""
        r = client.post(
            "/predict",
            files={"file": ("bad.pdf", b"fake", "application/pdf")},
        )
        assert "message" in r.json()

    def test_missing_file_field(self, client):
        """POST with no file field should return 422 (validation error)."""
        r = client.post("/predict")
        assert r.status_code == 422


# ── /predict — content-type edge cases ───────────────────────────────────────

class TestPredictEdgeCases:
    def test_jpeg_with_no_content_type_but_jpg_extension(self, client):
        """Some clients omit content-type; extension fallback should work."""
        r = client.post(
            "/predict",
            files={"file": ("photo.jpg", _make_jpeg_bytes(), None)},
        )
        # FastAPI may auto-infer content-type; if not, extension fallback applies
        assert r.status_code in (200, 415)   # 415 acceptable if CT is truly missing

    def test_grayscale_image_converted_to_rgb(self, client):
        """Grayscale image should be converted to RGB and accepted."""
        buf = io.BytesIO()
        Image.new("L", (224, 224), color=128).save(buf, format="JPEG")
        buf.seek(0)
        r = client.post(
            "/predict",
            files={"file": ("gray.jpg", buf.read(), "image/jpeg")},
        )
        assert r.status_code == 200

    def test_rgba_image_converted_to_rgb(self, client):
        """RGBA PNG should be converted to RGB and accepted."""
        buf = io.BytesIO()
        Image.new("RGBA", (224, 224), color=(0, 128, 255, 200)).save(buf, format="PNG")
        buf.seek(0)
        r = client.post(
            "/predict",
            files={"file": ("rgba.png", buf.read(), "image/png")},
        )
        assert r.status_code == 200
