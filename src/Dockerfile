# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Use a full Python image to install dependencies with compiled wheels.
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a prefix directory so we can copy them
# cleanly into the final stage without the full build toolchain.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Roy Alia Asiku <amroy776@gmail.com>"
LABEL description="RTV Field Image Classifier API"
LABEL version="1.0.0"

# Runtime system deps (Pillow needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy source files
# dataset.py is imported by api.py for the transform pipeline constants;
# model.py is imported for _build_model.
COPY dataset.py  dataset.py
COPY model.py    model.py
COPY api.py      api.py

# Checkpoint is mounted at runtime via volume or COPY in production.
# The default path can be overridden with the CHECKPOINT_PATH env var.
ENV CHECKPOINT_PATH=/app/checkpoints/best_model.pth

# Expose API port
EXPOSE 8000

# Non-root user for security
RUN useradd --create-home --no-log-init appuser
USER appuser

# Health check — Docker will mark the container unhealthy if /health fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Run with uvicorn. Workers=1 keeps a single model instance in memory.
# For multi-core scaling, use gunicorn with uvicorn workers (see Architecture Notes).
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
