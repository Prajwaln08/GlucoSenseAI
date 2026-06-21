# 🩸 GlucoSense AI

**Single-user, web-only, self-service blood-glucose prediction platform.**
One repo, one role (the end user), one deploy. Consolidates the former HealthGPT
backend (FastAPI + ML) into a single product — no doctor portal, no mobile app.

> Full blueprint: [docs/implementation_plan.md](docs/implementation_plan.md).

## What it does

- Register / login (JWT).
- Connect data sources:
  - **Junction** — **primary** CGM/data fetch (FreeStyle Libre + others).
  - **xDRIP+** — **fallback** CGM path, used automatically if Junction fails.
  - **Google Fit** — **all** Huawei-watch data (HR, steps, sleep, SpO₂, …).
- Log food; predict BGL (2 h / 3 h) with hypo/hyper + Clarke-zone risk flags.
- See history & trends, and which CGM source is currently live.
- Export / delete your data (privacy).

> The **AI chat/coach** (food & activity recommendations) is **deferred** to a
> future phase — see [`src/coach/`](src/coach/README.md).

## Architecture

```
src/
  api/          FastAPI app + routers (auth, predict, food, cgm, wearable, account)
  integrations/ unified ingestion: junction · xdrip · googlefit · ingest · cgm_router
  web/          server-rendered patient UI (Jinja2 + HTMX + Tailwind)
  models/ features/ serving/ data/ experiments/   ML stack (ported as-is)
  db/           SQLAlchemy models + Alembic migrations
  tasks/        Celery retrain + scheduled Junction sync
  coach/        ⏳ reserved stub for the future AI coach (not built)
  config.py     keeps the _CGMACROS_SKIP demo-user reservation (001/002)
```

## Local setup

```bash
conda activate glucosenseai            # env with deps (Python 3.11)
cp .env.example .env                   # fill DATABASE_URL, JUNCTION_*, GOOGLE_* ...
python -m alembic upgrade head         # apply migrations
uvicorn src.api.main:app --reload      # API + docs at /docs
```

## Security & operations

- **Auth:** JWT in an HttpOnly cookie (web) or `Authorization: Bearer` (API). Login is rate-limited; the xDRIP push needs a per-user key.
- **Secrets at rest:** the Google Fit refresh token is encrypted (Fernet) via `FIELD_ENCRYPTION_KEY` — generate one with
  `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.
- **CORS:** locked to `CORS_ORIGINS` (no `*` with credentials).
- **Privacy:** `GET /account/export` (download all data) and `DELETE /account` (erase).
- **Observability:** Prometheus metrics at `/metrics` — HTTP metrics plus CGM ingest by source, activity upserts by provider, and Junction API latency.
- **Deploy:** `render.yaml` (API + Celery worker w/ beat + Redis) or `docker-compose up`. The worker runs the scheduled Junction pull (`CGM_POLL_INTERVAL_SEC`).

## Status

Staged build (see the implementation plan):
- **Phase 0** — repo scaffold ✅
- **Phase 1** — remove doctor layer ✅
- **Phase 2** — patient web UI ✅
- **Phase 3** — Junction primary + xDRIP fallback (unified ingestion) ✅
- **Phase 4** — Google Fit watch sync ✅
- **Phase 5** — privacy / observability / deploy ✅
- **Phase 6** — *(future)* AI coach
