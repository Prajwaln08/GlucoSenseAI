# ──────────────────────────────────────────────────────────────────────────────
# GlucoSense AI — Training Image (Phase 1)
# Builds the full ML environment used for population & individual model training.
#
# Build:
#   docker build -f docker/Dockerfile -t glucosense-train:latest .
#
# A separate lighter Dockerfile.serve will be added in Phase 2 (FastAPI only,
# no torch/neuralprophet).
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# ── System deps ───────────────────────────────────────────────────────────────
# libgomp1  — required by LightGBM / XGBoost OpenMP threading
# git       — pip installs some packages directly from git refs
# curl      — used by the compose healthcheck on the MLflow service
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps — own layer so they cache unless requirements.txt changes ─────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
# data/, models/, reports/, and mlflow/ are bind-mounted at runtime — never baked in.
COPY src/       src/
COPY scripts/   scripts/
COPY pytest.ini .

# ── Runtime env ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Suppress the OpenMP duplicate-library warning on macOS hosts
    KMP_DUPLICATE_LIB_OK=TRUE

# Default to an interactive shell; override with the actual training command
# via `docker-compose run --rm train python scripts/train_population.py ...`
CMD ["bash"]
