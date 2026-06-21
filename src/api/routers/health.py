import json

from fastapi import APIRouter
from pydantic import BaseModel

from src.config import MODELS_DIR

router = APIRouter(tags=["health"])

REGISTRY_PATH = MODELS_DIR / "registry.json"


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version="2.0.0")


@router.get("/db-health")
def db_health() -> dict:
    """Check DB connectivity. Safe to expose — returns error text, no credentials."""
    from src.db.session import SessionLocal, DATABASE_URL
    masked = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return {"status": "ok", "host": masked}
    except Exception as exc:
        return {"status": "error", "host": masked, "detail": str(exc)}


@router.get("/models")
def list_models() -> dict:
    """List all registered models with key metrics."""
    if not REGISTRY_PATH.exists():
        return {"population": {}, "individual": {}}

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    return {
        "population": {
            dataset: {
                horizon: {
                    "model_type":   info["best_model_type"],
                    "test_rmse":    info["test_rmse"],
                    "clarke_a_pct": info["clarke_a_pct"],
                    "trained_at":   info["trained_at"],
                }
                for horizon, info in horizons.items()
            }
            for dataset, horizons in registry.get("population", {}).items()
        },
        "individual": {
            key: {
                "user_id":      info["user_id"],
                "dataset":      info["dataset"],
                "model_type":   info["model_type"],
                "horizon":      info["horizon"],
                "test_rmse":    info["test_rmse"],
                "clarke_a_pct": info["clarke_a_pct"],
            }
            for key, info in registry.get("individual", {}).items()
        },
    }
