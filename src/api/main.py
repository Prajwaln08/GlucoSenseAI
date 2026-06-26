"""
GlucoSense AI — FastAPI serving layer (Phase 2).
"""

# Load .env before any src module is imported — env vars must be set
# before auth/deps read os.environ.get(...) at module level.
from dotenv import load_dotenv
load_dotenv()

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.routers import account, auth, cgm, coach, food, googlefit, health, mobile, predict, wearable
from src.web.views import router as web_router
from src.serving.model_loader import warm_cache
from src.utils import get_logger

log = get_logger(__name__)

TAGS_METADATA = [
    {
        "name": "health",
        "description": "Liveness check and registered model registry. No auth required.",
    },
    {
        "name": "auth",
        "description": (
            "User registration and JWT authentication. "
            "Use **POST /auth/token** to get a Bearer token, then click **Authorize** above."
        ),
    },
    {
        "name": "predict",
        "description": (
            "Blood glucose prediction (2 h and 3 h horizons). "
            "Supports population models (CGMacros / Nature's Paper) and per-user individual models. "
            "Authentication is optional — if a valid token is provided the prediction is logged and drift detection runs."
        ),
    },
    {
        "name": "food",
        "description": "Meal logging — create, list, and delete food log entries. Requires authentication.",
    },
    {
        "name": "wearable",
        "description": (
            "Junction wearable integration. "
            "Connect 300+ devices (Dexcom, Fitbit, Garmin, Oura, Apple Health, etc.). "
            "Use **POST /wearable/link-token** to get a device-connection URL, "
            "**POST /wearable/sync** to pull the latest data, "
            "and **POST /wearable/webhook** for real-time Junction push events."
        ),
    },
    {
        "name": "cgm",
        "description": (
            "CGM ingestion + failover. Junction is the **primary** source (see *wearable*); "
            "xDRIP+ is the **fallback** via **POST /cgm/reading** (now authenticated with a "
            "per-user key — provision via **POST /cgm/key**). **GET /cgm/status** shows which "
            "source is live; **GET /cgm/readings** returns recent readings."
        ),
    },
    {
        "name": "googlefit",
        "description": (
            "Google Fit — the **sole source** for all Huawei-watch data (HR, steps, sleep, "
            "SpO₂, distance, calories). OAuth connect via **GET /googlefit/authorize**; pull "
            "days with **POST /googlefit/sync**."
        ),
    },
    {
        "name": "account",
        "description": "Privacy: **GET /account/export** (download all my data) and **DELETE /account** (erase).",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("GlucoSense AI API starting — warming model cache …")
    warm_cache()
    log.info("Model cache warm. Ready to serve predictions.")
    yield
    log.info("GlucoSense AI API shutting down.")


app = FastAPI(
    title="GlucoSense AI",
    description="""
## Blood Glucose Prediction API

Dual-model strategy: **population** (CGMacros / Nature's Paper datasets) + **individual** per-user models.
Prediction horizons: **2 h** and **3 h** ahead.

### Quick start

1. **Register** a patient account → `POST /auth/register`
2. **Login** → `POST /auth/token` (use the form, copy the `access_token`)
3. Click **Authorize** (top-right) and paste `Bearer <token>`
4. **Predict** → `POST /predict`
5. **Log food** → `POST /food/log`

### Datasets

| Dataset | Wearable | CGM | Users |
|---------|----------|-----|-------|
| `cgmacros` | FitBit Sense | FreeStyle Libre | 45 |
| `nature_paper` | Empatica E4 | Dexcom G6 | 7 |

### Model performance (population)

| Dataset | Horizon | Model | Test RMSE | Clarke A |
|---------|---------|-------|-----------|---------|
| nature_paper | 2h | XGBoost | 15.6 mg/dL | 98.0% |
| nature_paper | 3h | XGBoost | 17.7 mg/dL | 97.9% |
| cgmacros | 2h | LightGBM | 18.5 mg/dL | 96.2% |
| cgmacros | 3h | XGBoost | 21.1 mg/dL | 95.4% |
""",
    version="2.0.0",
    openapi_tags=TAGS_METADATA,
    contact={
        "name": "GlucoSense AI",
        "email": "prajwalnerkar01@gmail.com",
    },
    license_info={
        "name": "MIT",
    },
    swagger_ui_parameters={
        "persistAuthorization": True,
        "displayRequestDuration": True,
        "filter": True,
        "tryItOutEnabled": True,
    },
    lifespan=lifespan,
)

# Tightened CORS: explicit origins (required once cookies/credentials are allowed —
# "*" is invalid with allow_credentials=True). Set CORS_ORIGINS in .env (comma-sep).
_cors_origins = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── JSON API routers ──────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(predict.router, prefix="/predict")
app.include_router(food.router)
app.include_router(wearable.router)
app.include_router(cgm.router)
app.include_router(googlefit.router)
app.include_router(account.router)
app.include_router(mobile.router)
app.include_router(coach.router)

# ── Server-rendered patient web UI + static assets ────────────────────────────
_static_dir = Path(__file__).parent.parent / "web" / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
app.include_router(web_router)

# ── Observability: Prometheus HTTP metrics + domain counters at /metrics ──────
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})
