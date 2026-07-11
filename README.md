# GlucoSense AI

**An end-to-end blood-glucose forecasting platform** — raw CGM & wearable data in, personalized
glucose forecasts and an AI health assistant out. Shipped as an Android app, a web app, and a
public REST API.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi&logoColor=white)
![React Native](https://img.shields.io/badge/React_Native-Expo_SDK_56-61DAFB?logo=react&logoColor=black)
![CockroachDB](https://img.shields.io/badge/CockroachDB-Cloud-6933FF?logo=cockroachlabs&logoColor=white)
![Deploy](https://img.shields.io/badge/Deploy-Render_(Docker)-46E3B7?logo=render&logoColor=black)
![Tests](https://img.shields.io/badge/tests-145_passing-brightgreen)

---

## Table of Contents
1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Science Core](#3-data-science-core)
4. [Model Performance](#4-model-performance)
5. [Personalization Lifecycle](#5-personalization-lifecycle)
6. [AI Assistant — Doctor Gluco](#6-ai-assistant--doctor-gluco)
7. [Product Surfaces](#7-product-surfaces)
8. [Technology Stack](#8-technology-stack)
9. [Getting Started](#9-getting-started)
10. [Security & Privacy](#10-security--privacy)
11. [Roadmap](#11-roadmap)

---

## 1. Overview

GlucoSense AI ingests a person's real health signals from three independent sources — a CGM
sensor (xDRIP+ push or Junction webhooks), a smartwatch (Health Connect, per-minute vitals),
and food/vitals logs — and turns them into **short-horizon glucose forecasts (30–120 min)**
using a tiered model system that graduates from population models to models trained on the
individual.

Three product principles shape the system:

| Principle | Consequence in the product |
|---|---|
| **No fabricated values** | The app never renders demo, mock, or placeholder health data — empty states are honest. |
| **Forecasts must be grounded** | A *watch gate* blocks any prediction unless fresh heart-rate data is flowing (≥8 readings / 3 h, ≤30 min stale). |
| **Train ≡ Serve** | One shared feature pipeline builds both training sets and live serving frames — no train/serve skew by construction. |

---

## 2. System Architecture

```
  Watch (Health Connect) ──┐
  CGM  (xDRIP+ push) ──────┤   FastAPI ingest layer
  CGM  (Junction webhooks)─┼──►  · source-agnostic dedup (idempotent)
  Food & vitals (app/chat)─┘   · serialization-retry (CockroachDB)
                                      │
                                      ▼
                          CockroachDB Cloud (single store)
                                      │
              ┌───────────────────────┼───────────────────────────┐
              ▼                       ▼                           ▼
   Feature pipeline          Personalization scheduler      Doctor Gluco
   step1…step5 → 10-min      (Celery beat: sessions,        rules → facts
   grid, ~94 features        day-8 training, handovers)     LLM → phrasing
              │                       │                           │
              ▼                       ▼                           │
   Population models          Personal models                     │
   (4 horizons × 7 families)  (per user, per phase)               │
              └───────────────────────┴───────────────────────────┘
                                      ▼
                    Android app  ·  Web app (/app)  ·  REST API
                              (Render, Docker, us-east)
```

**Ingestion paths** are redundant by design: real-time webhooks (HMAC-verified), scheduled
pulls (Celery beat), manual sync, and batch backfill (`POST /cgm/entries`, Nightscout-style)
all funnel through one deduplicated write path.

---

## 3. Data Science Core

### 3.1 Datasets
| Source | Subjects | Contents |
|---|---|---|
| CGMacros | 45 | CGM traces, meal macronutrients, activity, demographics (~20 GB raw) |
| Nature-paper cohort | 9 | CGM + biomarkers/demographics |
| Live platform data | growing | Per-minute watch vitals, CGM streams, food logs — same schema as training |

### 3.2 Feature pipeline
`step1 load → step2 clinical-range checks + 10-minute grid → step3 imputation →
step4 feature build → step5 selection`, yielding **~94 features**: glucose lags and rolling
statistics, heart-rate rolling windows, meal-macro decay windows, circadian encodings, and
demographics. The identical pipeline (`build_live_frame → featurize`) constructs serving
frames from live database rows.

### 3.3 Model matrix
| Axis | Values |
|---|---|
| **Phase** | `while_on_cgm` · `post_cgm` (sensor gap) · `without_cgm` (watch-only "Virtual CGM") |
| **Tier** | population (cross-subject) · personal (per-user) |
| **Horizon** | 30 · 60 · 90 · 120 minutes |
| **Families** | ridge · XGBoost · LightGBM · CatBoost · HistGBR · ExtraTrees · MLP |

A model **registry** records metrics, versions, and artefacts per slot; serving loads the
winning family per horizon at runtime.

---

## 4. Model Performance

Population tier, held-out test on CGMacros (`while_on_cgm`):

| Horizon | RMSE (mg/dL) | MAE (mg/dL) | Clarke zone A |
|---:|---:|---:|---:|
| 30 min | **11.40** | 7.84 | 95.2 % |
| 60 min | 18.44 | 12.60 | 84.8 % |
| 90 min | 21.76 | 14.86 | 79.7 % |
| 120 min | 23.58 | 16.19 | 77.0 % |

Personal models are trained for 42 subjects (while-on-CGM) and 22 subjects (post-CGM)
across all four horizons. See §11 for the evaluation work in progress (baselines, MARD,
uncertainty quantification).

---

## 5. Personalization Lifecycle

Fully automated via Celery beat — no manual steps from first reading to personal forecasts:

```
first CGM reading ──► session opens ──► COLLECTING (population never shown as personal)
        day 8 ──► personal while_on_cgm model trains ──► PERSONAL serving
        day 13 ──► post_cgm model pre-trains (zero-gap handover)
  sensor silent 2h / day 14 ──► session ends ──► POST-CGM personal serving
  no sensor at all ──► "Virtual CGM": population watch+food model, clearly labeled estimates
```

Every phase is gated by live watch vitals; training additionally requires ≥30 % journey
watch coverage.

---

## 6. AI Assistant — "Doctor Gluco"

A grounded, guardrailed LLM assistant — **runs keyless on a local open-source model**
(Ollama / qwen2.5-7B) with a provider fallback chain (Anthropic → Ollama → deterministic
rules) so chat degrades gracefully rather than failing.

- **Grounded:** a structured context snapshot (the user's real glucose statistics, meals,
  vitals, activity) is the only permitted source of numbers.
- **Agentic:** tool-calling (`log_food`, `log_vitals`, `list_logs`) reads/writes the same
  database as the app; a once-per-turn execution guard prevents duplicate actions.
- **Guardrailed:** hard topic scope (diabetes, nutrition, exercise, sleep, wellbeing);
  safety rules (no dosing advice, emergencies escalate); polite refusal otherwise.
- **Fast:** SSE streaming (~1.4 s to first token on-device).
- **Suggestion ladder:** deterministic rules choose *what* to say (safety → forecast-reactive
  → data-gap nudges → insights); the LLM only rephrases, and any rewrite that drops a
  number is rejected by a fact guard.

---

## 7. Product Surfaces

| Surface | Highlights |
|---|---|
| **Android app** (React Native / Expo) | Live glucose graph (smooth, scrollable, food markers, local-time axis), Personalized vs Virtual-CGM modes with honest gating, watch-gate countdown timer, streaming chat, background Health Connect sync (15-min WorkManager + 5-min foreground), 30-day sessions with offline-tolerant restore |
| **Web app** (`/app`) | Single-page app consuming the same JSON API — home graph, streaming chat, profile |
| **REST API** | OpenAPI at `/docs`; per-user CGM push keys; Prometheus metrics at `/metrics` |

---

## 8. Technology Stack

| Layer | Technology |
|---|---|
| ML / Data | pandas, scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, MLflow |
| Backend | FastAPI, SQLAlchemy + Alembic, Celery (beat) + Redis, Pydantic |
| LLM | Ollama (qwen2.5-7B) with tool-calling, SSE streaming; optional Anthropic |
| Database | CockroachDB Cloud (serializable; retry-aware ingest) |
| Mobile | React Native, Expo SDK 56, TanStack Query, react-native-health-connect |
| Deploy | Render (Docker, region-pinned to DB), EAS builds, docker-compose for local |

---

## 9. Getting Started

```bash
# Backend (full local stack: API + Celery beat + Redis + MLflow)
conda activate glucosenseai          # Python 3.10
cp .env.example .env                 # DATABASE_URL, SECRET_KEY, optional JUNCTION_*
make api && make worker

# Tests
pytest tests -q                      # 145 tests

# Mobile
cd mobile && npx expo start --dev-client    # EAS build profiles in mobile/eas.json
```

The API serves docs at `http://localhost:8000/docs` and the web app at
`http://localhost:8000/app`.

---

## 10. Security & Privacy

- JWT auth — 30-day tokens with a sliding 5-day idle logout enforced client-side; only an
  explicit `401` terminates a session (network failures never do).
- Per-user CGM push keys (rotatable); webhook HMAC-SHA256 verification (Svix format).
- Secrets encrypted at rest (Fernet); CORS locked; login rate-limited.
- Full data export and account deletion endpoints.
- **No demo or fabricated values anywhere in the product.**

---

## 11. Roadmap

**Evaluation rigor (in progress):** naive-baseline reporting (persistence / trend),
MARD + full error-grid distributions, hypo/hyper event-detection metrics, conformal
prediction intervals, leakage-audited subject-wise CV, segmented error analysis,
paired statistical tests for personalization lift, drift monitoring.

**Platform:** model hosting off-device, Play Store release (internal → closed → production),
Junction production connector (currently sandbox-only), iOS.
