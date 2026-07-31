# GlucoSense AI

**An end-to-end blood-glucose forecasting platform** — raw CGM & wearable data in,
personalized glucose forecasts and an AI health assistant out. Built as a complete data
science project: data engineering → feature pipeline → a tiered model system → rigorous
offline evaluation → a deployed product (Android app + web app + public REST API) with an
LLM health assistant on top.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-·_XGBoost_·_LightGBM_·_CatBoost-F7931E?logo=scikitlearn&logoColor=white)
![React Native](https://img.shields.io/badge/React_Native-Expo_SDK_56-61DAFB?logo=react&logoColor=black)
![CockroachDB](https://img.shields.io/badge/CockroachDB-Cloud-6933FF?logo=cockroachlabs&logoColor=white)
![Deploy](https://img.shields.io/badge/Deploy-Render_(Docker)-46E3B7?logo=render&logoColor=black)
![Tests](https://img.shields.io/badge/tests-145_passing-brightgreen)

> **Scope note.** This is a **research & portfolio project**, distributed as a **direct-install
> Android APK** for testing — it is **not** published to the Play Store and is **not a medical
> device**. Forecasts are educational, not clinical advice.

---

## Table of Contents
1. [Try It](#1-try-it)
2. [Overview](#2-overview)
3. [System Architecture](#3-system-architecture)
4. [Data](#4-data)
5. [The Three Forecasting Phases](#5-the-three-forecasting-phases)  ← *the model types*
6. [Feature Pipeline](#6-feature-pipeline)
7. [Model Performance & Evaluation](#7-model-performance--evaluation)  ← *the DS core*
8. [Personalization Lifecycle](#8-personalization-lifecycle)
9. [AI Assistant — Doctor Gluco](#9-ai-assistant--doctor-gluco)
10. [Product Surfaces](#10-product-surfaces)
11. [Technology Stack](#11-technology-stack)
12. [Getting Started](#12-getting-started)
13. [Security & Privacy](#13-security--privacy)
14. [Project Status & Scope](#14-project-status--scope)

---

## 1. Try It

| What | Link |
|---|---|
| 📱 **Android app (APK)** | `<!-- TODO: paste the EAS APK link here -->` — open on Android, allow install from browser |
| 🌐 **Web app** (same account & data, any browser) | https://glucosense-api-dj43.onrender.com |
| 🔌 **REST API docs** (OpenAPI) | https://glucosense-api-dj43.onrender.com/docs |

**Quick start in the app:** register with an email → fill the short health profile →
connect a data source (a smartwatch via Health Connect for "Virtual CGM" estimates, and/or
a CGM sensor). No data is fabricated — empty states are honest until real data flows.

> The public backend runs on a **free tier that sleeps after ~15 min idle**, so the first
> request after a quiet spell takes ~30–60 s to wake. That is expected, not a bug.

---

## 2. Overview

GlucoSense AI ingests a person's real health signals from three independent sources — a CGM
sensor (xDRIP+ push or Junction webhooks), a smartwatch (Health Connect, per-minute vitals),
and food/vitals logs — and turns them into **short-horizon glucose forecasts (30 / 60 / 90 /
120 min)** using a tiered model system that graduates from population models to models trained
on the individual.

Three engineering principles shape the system:

| Principle | Consequence |
|---|---|
| **No fabricated values** | The product never renders demo, mock, or placeholder health data. |
| **Forecasts must be grounded** | A *watch gate* blocks any prediction without fresh heart-rate data (≥8 readings / 3 h, ≤30 min stale). |
| **Train ≡ Serve** | One shared feature pipeline builds both training sets and live serving frames — no train/serve skew by construction. |

---

## 3. System Architecture

```
  Watch (Health Connect) ──┐
  CGM  (xDRIP+ push) ──────┤   FastAPI ingest layer
  CGM  (Junction webhooks)─┼──►  · source-agnostic dedup (idempotent)
  Food & vitals (app/chat)─┘   · CockroachDB serialization-retry
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
```

Ingestion is redundant by design: real-time webhooks (HMAC-verified), scheduled Celery pulls,
manual sync, and batch backfill — all through one deduplicated, retry-aware write path.

---

## 4. Data

Two research cohorts provide the training data; live users add their own once deployed.

| Source | Subjects | Contents |
|---|---:|---|
| **CGMacros** | 45 | CGM traces, meal macronutrients, activity, demographics (~20 GB raw) |
| **Nature-paper cohort** | 9 | CGM + biomarkers / demographics |
| **Live platform data** | growing | Per-minute watch vitals (HR, steps, SpO₂, calories), CGM streams, food logs — same schema the models train on |

<!-- TODO (data provenance — fill in): -->
<!--  · CGMacros dataset — <platform/dataset name + URL + license> -->
<!--  · Nature-paper cohort — <paper title, authors, journal, year, DOI/URL> -->
> **Data sources.** The two research cohorts come from a published dataset platform and a
> peer-reviewed research paper *(citations to be added above)*. Raw data is **gitignored** and
> never committed; only trained-model metadata and aggregate metrics live in the repo.

**Preprocessing target grid:** every subject is resampled to a **10-minute grid**, clinically
range-checked, and imputed — producing ~**94 features** and forecast targets at 30/60/90/120 min.

---

## 5. The Three Forecasting Phases

The heart of the system: glucose forecasting is not one model but **three phases**, matched to
what data the user has *right now*. Each phase has its own target formulation, split, and model
matrix, and the app serves whichever phase the user is in.

| Phase | When it applies | Inputs | Target | Train/Val/Test (days) |
|---|---|---|---|---|
| **1. While-on-CGM** | A CGM sensor is actively streaming | glucose + watch + food | **Δ** glucose vs. now (anchored to current reading) | 6 / 2 / 0¹ |
| **2. Post-CGM** | Sensor session ended (gap between sensors) | watch + food (+ recent glucose history) | **absolute** glucose | 8 / 2 / 2 |
| **3. Virtual CGM** *(without-CGM)* | User has **no** sensor — a watch only | watch + food + demographics | **absolute** glucose | 7 / 2 / 2 |

¹ While-on-CGM keeps no held-out test split — the *live CGM stream is the real test_, so models
are validated on the last 2 days. (See [validation.md](docs/validation.md).)

**How each phase works**

- **While-on-CGM** predicts a **delta** off the current reading (`ŷ = current + Δ̂`). Because
  the last glucose value is such a strong anchor, delta-modeling lets the model focus on the
  *change*, which is where the signal is. This is the primary, most-accurate phase.
- **Post-CGM** bridges the gap when one sensor ends and the next hasn't started: no live
  glucose anchor, so it predicts **absolute** glucose from watch + food + the user's recent
  history, and is pre-trained at day 13 for a zero-gap handover.
- **Virtual CGM** is the "no hardware" experience: it estimates absolute glucose from a
  **smartwatch and food logs alone** — clearly labeled as an *estimate*, gated on live heart
  rate, so a watch-only user still gets forecasts.

Across all three phases the app serves the **winning model family per (phase, horizon)** from a
model registry (7 families evaluated: ridge · XGBoost · LightGBM · CatBoost · HistGBR ·
ExtraTrees · MLP), at **two tiers** — *population* (cross-subject) and *personal* (per user,
trained once enough of their own data exists; see §8).

---

## 6. Feature Pipeline

`step1 load → step2 clinical-range checks + 10-minute grid → step3 imputation →
step4 feature build → step5 selection`, producing ~**94 features**:

- **Glucose dynamics** — lags (10/30/60/120/180 min), rolling mean/std, deltas, rate-of-change
- **Heart-rate** — rolling windows over per-minute watch data
- **Meals** — macro (carb/protein/fat/fiber/sugar) decay windows at 1/2/3 h
- **Context** — circadian (hour-of-day) encodings, demographics (age, BMI, HbA1c, sex)

The **identical** pipeline (`build_live_frame → featurize`) builds live serving frames from
database rows, so a model trains and serves on the same features — **no train/serve skew**. All
features are strictly **backward-looking** (audited — see §7 leakage audit).

---

## 7. Model Performance & Evaluation

Population tier, CGMacros While-on-CGM (45 subjects, ~12k validation samples). The evaluation is
deliberately **self-critical** — it reports where the models fail, not just where they win.
Everything here is reproducible from `scripts/evaluate_*.py`, run **offline** (never imported by
the app; 145 unit tests remain green).

### 7.1 Accuracy vs. a naive baseline *(is it better than "predict the last reading"?)*
| Horizon | Persistence RMSE | Best model RMSE | Improvement | MARD | Clarke A |
|---:|---:|---:|---:|---:|---:|
| 30 min | 16.38 | **11.36** (catboost) | **31 %** | 7.3 % | 95.2 % |
| 60 min | 26.40 | **18.35** | **31 %** | 11.6 % | 84.5 % |
| 90 min | 32.9 | **21.72** | **34 %** | 12.4 % | 79.7 % |
| 120 min | ~36 | **23.58** | ~30 % | 16.2 % | 77.0 % |

Models beat persistence at every horizon; a linear-trend baseline does *worse* than persistence
(noise amplification) — reported as an honest negative result.

### 7.2 Clinical utility — RMSE ≠ safety
Treating each forecast as a **hypoglycemia (<70) detector**: at 120 min the *lowest-RMSE* models
catch **fewer** dangerous lows (sensitivity 0.43–0.50) than **naive persistence (0.63)**. They
minimize squared error by regressing toward the mean, which blinds them to the extremes.
Segmented error confirms it — severe-hypo MARD (15.8 %) ≈ 2.4× the in-range MARD (6.6 %).

### 7.3 Uncertainty — conformal prediction intervals
Split-conformal gives **92 % empirical coverage at a 90 % target**, with width honestly growing
±19 → ±39 mg/dL over 30 → 120 min. Conditional coverage exposes the catch: severe-hyperglycemia
is covered only **22 %** of the time — the marginal guarantee hides a tail failure (fix:
Mondrian per-band conformal).

### 7.4 Personalization lift — does per-user training help? *(paired Wilcoxon, 42 subjects)*
Personalization is **null at 30 min** but its value **grows with horizon**: by 120 min personal
models significantly beat population on **MARD (+5.2 %, p = 0.036)** and **hypo-detection
(+15 pp, p = 0.030, medium effect dz = 0.48)**. Knowing *when* a technique helps — and being
honest when it doesn't — is the point.

### 7.5 Generalization — leave-one-subject-out CV *(the cold-start penalty)*
A brand-new user (unseen subject) pays a penalty that **grows monotonically with horizon**:
+13.7 % (30 m) → +24.2 % (120 m) RMSE, all p < 1e-8. This is the honest deployment number the
in-sample split hides — and it's exactly the gap the personalization lifecycle (§8) is designed
to close.

### 7.6 Calibration & drift
- **Calibration:** well-calibrated overall (ECE 0.7–1.8 mg/dL), but **under-predicts highs**
  (top-decile bias −7.5 mg/dL at 120 min) — the same mean-regression weakness seen in §7.2/§7.3.
- **Drift monitoring:** PSI(train vs. live glucose) = **0.50 → significant shift**; the live
  population runs ~3× more hypoglycemic than the training cohort. A concrete monitoring plan
  (nightly PSI alerts, rolling live-RMSE, retrain triggers) is documented.

### 7.7 Validation & leakage audit
Splits are **temporal, per-subject, deterministic**; every fitted transform (selection, imputer,
scaler) is **fit on train only**; all features are **backward-shifted** (verified — targets look
forward, features never do). Full write-up: **[docs/validation.md](docs/validation.md)**.

> **One-line evaluation story:** *models beat baselines 31–34 %, but RMSE hides poor hypo-
> detection and under-predicted highs; intervals are calibrated marginally but under-cover the
> extremes; new users pay a 14–24 % cold-start penalty — biggest at long horizons — which
> personalization significantly closes; and the live population is 3× more hypoglycemic than
> training, exactly where the model is weakest.* Every finding names a limitation and its fix.

---

## 8. Personalization Lifecycle

Fully automated via Celery beat — no manual step from first reading to personal forecasts:

```
first CGM reading ──► session opens ──► COLLECTING (population model, honest label)
        day 8 ──► personal While-on-CGM model trains ──► PERSONAL serving
        day 13 ──► Post-CGM model pre-trains (zero-gap handover)
  sensor silent 2h / day 14 ──► session ends ──► POST-CGM personal serving
  no sensor at all ──► Virtual CGM (population watch+food model, labeled estimate)
```

Every phase is gated on live watch vitals; training additionally requires ≥30 % journey watch
coverage. This lifecycle is the direct answer to the cold-start penalty measured in §7.5.

---

## 9. AI Assistant — "Doctor Gluco"

A grounded, guardrailed LLM assistant that **runs keyless on a local open-source model**
(Ollama / qwen2.5-7B) with a provider fallback chain (Anthropic → Ollama → deterministic rules)
so chat degrades gracefully rather than failing.

- **Grounded** — a context snapshot of the user's real glucose/meals/vitals is the only source
  of numbers.
- **Agentic** — tool-calling (`log_food`, `log_vitals`, `list_logs`) reads/writes the same DB as
  the app; a once-per-turn guard prevents duplicate actions.
- **Guardrailed** — hard topic scope (diabetes, nutrition, exercise, sleep, wellbeing); safety
  rules (no dosing, emergencies escalate); polite refusal otherwise.
- **Fast** — SSE streaming (~1.4 s to first token on-device).
- **Suggestion ladder** — rules choose *what* to say (safety → forecast-reactive → data-gap →
  insight); the LLM rephrases, and a fact-guard rejects any rewrite that drops a number.

---

## 10. Product Surfaces

| Surface | Highlights |
|---|---|
| **Android app** (React Native / Expo) | Live glucose graph (smooth, scrollable, food markers, local-time axis); Personalized vs. Virtual-CGM modes with honest gating; watch-gate countdown; streaming chat; background Health Connect sync; 30-day sessions with offline-tolerant restore |
| **Web app** (`/app`) | Single-page app on the same JSON API — home graph, streaming chat, profile |
| **REST API** | OpenAPI at `/docs`; per-user CGM push keys; Prometheus metrics at `/metrics` |

---

## 11. Technology Stack

| Layer | Technology |
|---|---|
| ML / Data | pandas, scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, SciPy, MLflow |
| Backend | FastAPI, SQLAlchemy + Alembic, Celery (beat) + Redis, Pydantic |
| LLM | Ollama (qwen2.5-7B) with tool-calling + SSE streaming; optional Anthropic |
| Database | CockroachDB Cloud (serializable; retry-aware ingest) |
| Mobile | React Native, Expo SDK 56, TanStack Query, react-native-health-connect |
| Deploy | Render (Docker, region-pinned to DB), EAS builds, docker-compose for local |

---

## 12. Getting Started

```bash
# Backend — full local stack (API + Celery beat + Redis + MLflow)
conda activate glucosenseai            # Python 3.10
cp .env.example .env                   # DATABASE_URL, SECRET_KEY, optional JUNCTION_*
make api && make worker

# Tests (145)
pytest tests -q

# Offline evaluation (reproduces §7 — never touches the running app)
python scripts/evaluate_baselines.py            # baselines + clinical metrics
python scripts/evaluate_personalization.py      # personal vs population (paired Wilcoxon)
python scripts/evaluate_conformal.py            # prediction intervals + conditional coverage
python scripts/evaluate_loso.py                 # cold-start LOSO CV
python scripts/evaluate_calibration_segments.py # calibration + deeper segmentation
python scripts/evaluate_drift.py                # train→deploy PSI drift

# Mobile
cd mobile && npx expo start --dev-client        # build profiles in mobile/eas.json
```

API docs: `http://localhost:8000/docs` · web app: `http://localhost:8000/app`.

---

## 13. Security & Privacy

- JWT auth — 30-day tokens with a sliding 5-day idle logout; only an explicit `401` ends a
  session (network failures never do).
- Per-user CGM push keys (rotatable); webhook HMAC-SHA256 verification.
- Secrets encrypted at rest (Fernet); CORS locked; login rate-limited.
- Full data export and account deletion.
- **No demo or fabricated values anywhere in the product.**

---

## 14. Project Status & Scope

**Delivered end-to-end:** data engineering → feature pipeline → 3-phase × 2-tier × 4-horizon
model system → an 8-point rigorous evaluation (baselines, clinical metrics, uncertainty,
personalization stats, LOSO, calibration, drift, leakage audit) → deployed Android app, web app,
and public API, with a keyless LLM assistant.

**Intentionally out of scope:** this is a **portfolio / research project shipped as a testable
APK**, not a Play Store release and **not a medical device**. Forecasts are educational only.

**Roadmap (next):** hypo-weighted / asymmetric loss to fix the §7.2 sensitivity gap; Mondrian
conformal for the §7.3 tails; conformal bands surfaced on the app graph; model hosting fully
off-device.
