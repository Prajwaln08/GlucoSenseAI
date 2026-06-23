# ──────────────────────────────────────────────────────────────────────────────
# GlucoSense AI — Makefile
# ──────────────────────────────────────────────────────────────────────────────

COMPOSE        = docker compose
TRAIN_RUN      = $(COMPOSE) --profile train run --rm train
PYTHON         = python

.PHONY: help \
        build build-api mlflow mlflow-down \
        api api-down api-logs \
        worker worker-down \
        db-up db-migrate db-down \
        train-cgmacros train-np train-individual train-all \
        save-features \
        test lint \
        clean

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "GlucoSense AI — available targets"
	@echo ""
	@echo "  Infrastructure"
	@echo "    build              Build / rebuild the training Docker image"
	@echo "    build-api          Build / rebuild the serving Docker image"
	@echo "    mlflow             Start MLflow tracking server (http://localhost:5000)"
	@echo "    mlflow-down        Stop MLflow server"
	@echo "    api                Start postgres + redis + mlflow + API"
	@echo "    api-down           Stop prediction API"
	@echo "    api-logs           Tail API container logs"
	@echo "    worker             Start Celery retraining worker"
	@echo "    worker-down        Stop Celery worker"
	@echo ""
	@echo "  Database"
	@echo "    db-up              Start PostgreSQL (if not already running)"
	@echo "    db-migrate         Run Alembic migrations (alembic upgrade head)"
	@echo "    db-down            Stop PostgreSQL"
	@echo ""
	@echo "  Data"
	@echo "    save-features      Serialise feature matrices to parquet for both datasets"
	@echo ""
	@echo "  Training (inside Docker, logs to MLflow)"
	@echo "    train-cgmacros     Population model — CGMacros, 2h + 3h horizons"
	@echo "    train-np           Population model — Nature's Paper, 2h + 3h horizons"
	@echo "    train-individual   Best individual model for each dataset"
	@echo "    train-all          Run all four training jobs sequentially"
	@echo ""
	@echo "  Dev"
	@echo "    test               Run pytest suite"
	@echo "    lint               Run flake8 + isort check"
	@echo "    clean              Remove __pycache__ trees"
	@echo ""

# ── Infrastructure ────────────────────────────────────────────────────────────
build:
	$(COMPOSE) build train

build-api:
	$(COMPOSE) build api

mlflow:
	$(COMPOSE) up -d mlflow

mlflow-down:
	$(COMPOSE) stop mlflow

api:
	$(COMPOSE) up -d postgres redis mlflow api

api-down:
	$(COMPOSE) stop api

api-logs:
	$(COMPOSE) logs -f api

worker:
	$(COMPOSE) --profile worker up -d worker

worker-down:
	$(COMPOSE) stop worker

db-up:
	$(COMPOSE) up -d postgres

db-migrate:
	$(PYTHON) -m alembic upgrade head

db-down:
	$(COMPOSE) stop postgres

# ── Training (unified tier pipeline — see scripts/train.py) ───────────────────
train-cgm-active:
	$(TRAIN_RUN) python scripts/train.py --tier while_on_cgm

train-without-cgm:
	$(TRAIN_RUN) python scripts/train.py --tier without_cgm

train-post-cgm:
	$(TRAIN_RUN) python scripts/train.py --tier post_cgm

train-all: train-cgm-active train-without-cgm train-post-cgm

# ── Dev ───────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	flake8 src/ scripts/ --max-line-length=100
	isort --check-only src/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
