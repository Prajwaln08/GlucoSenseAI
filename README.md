# 🩸 GlucoSense AI

**End-to-end blood-glucose forecasting platform** — from raw CGM/wearable datasets to
trained personalized models to a deployed product (Android app + web app + public API)
with an LLM health assistant on top.

One person's real data flows in from three sources (CGM sensor, smartwatch, food logs),
through a versioned feature pipeline, into per-user forecast models whose lifecycle
(train → serve → retire → retrain) is fully automated.

```
Watch (Health Connect) ─┐
CGM (xDRIP+ / Junction) ─┼─► FastAPI ingest (deduped, source-agnostic)
Food & vitals (app/chat)─┘        │
                                  ▼
                       CockroachDB (single store)
                                  │
              ┌───────────────────┼──────────────────────┐
              ▼                   ▼                      ▼
     10-min feature pipeline   personalization      Doctor Gluco
     (grid→impute→94 feats)    lifecycle (Celery)   (LLM + tools + rules)
              │                   │                      │
              ▼                   ▼                      │
     tiered forecast models  personal models             │
     (population, 4 horizons) (per user, day 8+)         │
              └───────────────────┴──────────────────────┘
                                  ▼
                 Android app · Web app · REST API (Render)
```

## The data science core

### Data
- **CGMacros** — 45 subjects: CGM traces, meal macros, activity, demographics (~20 GB raw).
- **Nature-paper cohort** — 9 subjects with bio/demographics.
- **Live user data** — per-minute watch samples (HR, steps, SpO₂, calories), CGM streams,
  food logs with macros — the same schema the models train and serve on.

### Feature pipeline (train ≡ serve, no skew)
`step1 load → step2 10-min grid + clinical-range checks → step3 imputation → step4 features
→ step5 selection`, producing **~94 features**: glucose lags/rolls, HR rolling windows,
meal-macro decay windows, circadian encodings, demographics. The **identical code path**
builds live serving frames from DB rows (`build_live_frame → featurize`), so a personal
model trains and serves on the same features it will see in production.

### Model matrix
| Axis | Values |
|---|---|
| Phase | `while_on_cgm` (sensor live) · `post_cgm` (sensor gap) · `without_cgm` (watch-only "Virtual CGM") |
| Tier | population (cross-subject) · personal (per-user, trained at day 8 of their sensor session) |
| Horizon | 30 / 60 / 90 / 120 min |
| Families | ridge · xgboost · lightgbm · catboost · histgbr · extratrees · mlp — winner per slot via registry |

Current population headline (held-out test, CGMacros): **30-min RMSE ≈ 11.4 mg/dL
(MAE 7.8, Clarke zone-A 95.2%)** degrading to ≈ 23.6 mg/dL at 120 min. Personal models
trained for 42 study subjects (while-on-CGM) and 22 (post-CGM).

### Personalization lifecycle (automated)
CGM readings auto-open a **sensor session**; a Celery scheduler advances each user through
phases: collecting → day-8 personal training → personal serving → (sensor dies) →
post-CGM model, pre-trained at day 13 for a zero-gap handover. A **watch gate** (≥8 HR
readings in 3 h, ≤30-min staleness) blocks any forecast without live vitals — the product
never serves a guess it can't ground, and never shows fabricated values anywhere.

### Evaluation roadmap (known gaps — active work)
The next layer of rigor, in priority order:
1. **Naive baselines** (persistence / linear-trend) reported beside every model — RMSE
   without a baseline is uninterpretable at short horizons.
2. **Clinical metrics**: MARD, full Clarke/Parkes error-grid distributions, and
   **hypo/hyper event detection** (sensitivity/FPR/lead-time for crossing 70/180 mg/dL).
3. **Uncertainty**: quantile or conformal prediction intervals instead of point forecasts.
4. **Leakage-audited validation**: subject-wise (LOSO) CV for population models, strictly
   temporal splits for personal models, documented.
5. **Segmented error analysis**: hypo range vs euglycemia, post-meal vs fasting, night vs day.
6. **Statistical testing**: bootstrap CIs on metrics; paired tests for personal-vs-population.
7. **Drift monitoring**: study-data → real-device covariate shift, PSI-style alerts.

## The AI layer — "Doctor Gluco"
- **Keyless local LLM** (Ollama, qwen2.5-7B) with provider fallback (Anthropic → Ollama →
  deterministic rules) — chat never 500s.
- **Grounded**: a context snapshot (their real glucose stats, meals, vitals, activity) is
  the only source of numbers; tool-calling (`log_food`, `log_vitals`, `list_logs`) writes
  to and reads from the same DB as the app; a once-per-turn guard stops duplicate logging.
- **Guardrailed**: hard scope (diabetes/nutrition/exercise/sleep/wellbeing only), safety
  rules (no dosing, emergencies escalate), SSE **streaming** replies (~1.4 s first token).
- **Suggestion ladder**: deterministic rules pick WHAT to say (safety → forecast-reactive
  → data-gap nudges → insights); the LLM phrases HOW — with a fact-guard that rejects any
  rewrite that drops a number.

## Product & platform
- **Android app** (React Native / Expo): live glucose graph (smooth, scrollable, food
  markers), Personalized vs Virtual-CGM modes with honest gating, watch-gate countdown,
  streaming chat, background Health Connect sync (15-min WorkManager + 5-min foreground).
- **Web app** (`/app`): single-page, same JSON API as mobile — Home graph, streaming chat,
  profile.
- **Ingestion**: xDRIP+ push (single + batch backfill endpoints), Junction webhooks
  (HMAC-verified) + scheduled pulls, Health Connect per-minute samples; all deduped
  through one idempotent write path with CockroachDB serialization-retry.
- **Deploy**: Render (Docker, region-pinned next to the DB) + CockroachDB Cloud; hybrid
  worker/model-serving on a dev machine. `docker-compose` for full local stack. 145 tests.

## Local setup
```bash
conda activate glucosenseai              # Python 3.10 env
cp .env.example .env                     # DATABASE_URL, SECRET_KEY, (optional) JUNCTION_*
make api && make worker                  # full docker stack: API + Celery(beat) + Redis + MLflow
cd mobile && npx expo start --dev-client # app dev server (EAS builds via eas.json)
pytest tests -q                          # 145 tests
```

## Security & privacy
JWT auth (30-day tokens, 5-day idle logout client-side) · per-user CGM push keys ·
HMAC-verified webhooks · secrets encrypted at rest (Fernet) · Prometheus `/metrics` ·
full data export & account deletion · **no demo or fabricated values anywhere in the product**.
