"""
Celery task: trigger_retrain

Runs an individual model retraining job for a specific user.

Flow:
    1. Mark job as "running" in DB.
    2. Build user feature matrix — tries live DB data first (CgmReading +
       FoodLog + WearableActivity); falls back to parquet if DB has < MIN_ROWS_INDIVIDUAL rows.
    3. Run the full model zoo; pick the best model by val RMSE.
    4. Compare new test_rmse against the population registry RMSE for this
       dataset/horizon (and against the user's existing individual RMSE if one exists).
    5. Write the result to registry["individual"][dataset][user_id][horizon].
    6. Save artefact_dir + version_tag to the RetrainJob row.
    7. Invalidate the per-user model loader cache so the next prediction picks
       up the new model immediately.

Drift trigger (called from the API after prediction logging):
    If recent_prediction_rmse() > population registry RMSE * (1 + DRIFT_THRESHOLD),
    a retrain job is enqueued with triggered_by="auto_drift".
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.tasks.celery_app import celery_app
from src.utils import get_logger

log = get_logger(__name__)

DRIFT_THRESHOLD = float(os.environ.get("RMSE_DRIFT_ALERT_THRESHOLD_PCT", "20")) / 100.0
MAX_RETRAIN_QUEUE = int(os.environ.get("MAX_RETRAIN_QUEUE_SIZE", "5"))

_MEAL_TYPE_MAP = {
    "breakfast": 1, "lunch": 2, "dinner": 3,
    "snack": 4, "other": 0,
}


@celery_app.task(
    bind=True,
    name="src.tasks.retrain.trigger_retrain",
    max_retries=2,
    default_retry_delay=60,
)
def trigger_retrain(
    self,
    job_id: str,
    user_id: str,
    dataset: str,
    horizon: str,
) -> dict:
    """
    Retrain the individual model for one user across the full model zoo.

    Args:
        job_id:   RetrainJob.id — updated throughout with status, RMSE, artefact_dir.
        user_id:  Participant ID string (e.g. "019").
        dataset:  "cgmacros" | "nature_paper".
        horizon:  "2h" | "3h".

    Returns:
        dict with status, test_rmse, prev_rmse, improved, version_tag.
    """
    from src.config import DATA_PROCESSED_DIR, MODELS_DIR
    from src.db.crud import update_retrain_job
    from src.db.models import RetrainJob
    from src.db.session import SessionLocal
    from src.features.pipeline import build_feature_matrix

    db = SessionLocal()
    try:
        _mark(db, job_id, "running", started_at=datetime.now(timezone.utc),
              celery_id=self.request.id)

        # Resolve the internal DB PK for this user from the job row itself
        job = db.query(RetrainJob).filter(RetrainJob.id == job_id).first()
        if job is None:
            raise RuntimeError(f"RetrainJob {job_id} not found in DB.")
        user_pk = job.user_id_fk

        # ── 1. Build feature matrix ───────────────────────────────────────────
        df = None
        source = "unknown"

        try:
            df = _load_user_from_db(db, user_pk, user_id)
            source = "db"
            log.info(
                f"Loaded {len(df)} rows from DB for {dataset}/{user_id}"
            )
        except Exception as db_exc:
            log.warning(f"DB load failed for {dataset}/{user_id}: {db_exc} — trying parquet.")

        if df is None or len(df) < 200:
            parquet_path = DATA_PROCESSED_DIR / dataset / f"{user_id}.parquet"
            if parquet_path.exists():
                import pandas as pd
                df_raw = pd.read_parquet(parquet_path)
                df = build_feature_matrix(df_raw, user_id=user_id)
                source = "parquet"
                log.info(
                    f"Loaded {len(df)} rows from parquet for {dataset}/{user_id}"
                )
            else:
                rows = len(df) if df is not None else 0
                raise FileNotFoundError(
                    f"Insufficient data for {dataset}/{user_id}: "
                    f"{rows} DB rows and no parquet fallback."
                )

        if len(df) < 200:
            raise ValueError(
                f"Only {len(df)} rows after feature engineering (source={source}) — "
                "minimum 200 required."
            )

        log.info(
            f"Retraining {dataset}/{user_id}/{horizon} on {len(df)} rows (source={source})"
        )

        # ── 2. Run zoo — pick best model by val RMSE ─────────────────────────
        result, best_name = _run_zoo(df, dataset, user_id, horizon)

        new_rmse = result.test_rmse
        new_val  = result.val_rmse

        # ── 3. Compare against registry ───────────────────────────────────────
        registry_path = MODELS_DIR / "registry.json"
        prev_rmse  = _get_prev_rmse(registry_path, dataset, horizon, user_id)
        pop_rmse   = _get_population_rmse(registry_path, dataset, horizon)
        improved   = prev_rmse is None or new_rmse < prev_rmse

        # ── 4. Build version tag ──────────────────────────────────────────────
        version_tag = f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}_{job_id[:8]}"

        # ── 5. Write to registry ──────────────────────────────────────────────
        _update_registry_individual(
            registry_path=registry_path,
            dataset=dataset,
            horizon=horizon,
            user_id=user_id,
            model_type=best_name,
            artefact_dir=result.artefact_dir,
            val_rmse=new_val,
            test_rmse=new_rmse,
            clarke_a_pct=result.clarke_a_pct,
            version_tag=version_tag,
        )

        # ── 6. Invalidate per-user model cache ────────────────────────────────
        _invalidate_cache(dataset, horizon, user_id)

        # ── 7. Persist artefact_dir + version_tag to DB ───────────────────────
        _mark(
            db, job_id, "done",
            completed_at=datetime.now(timezone.utc),
            test_rmse=new_rmse,
            prev_rmse=prev_rmse,
            improved=improved,
            artefact_dir=str(result.artefact_dir),
            version_tag=version_tag,
        )

        log.info(
            f"Retrain done: {dataset}/{horizon} user={user_id} "
            f"model={best_name} RMSE {prev_rmse} → {new_rmse:.2f} "
            f"(pop={pop_rmse}) improved={improved} tag={version_tag}"
        )

        return {
            "status": "done",
            "test_rmse": new_rmse,
            "prev_rmse": prev_rmse,
            "improved": improved,
            "version_tag": version_tag,
            "source": source,
        }

    except Exception as exc:
        log.error(f"Retrain job {job_id} failed: {exc}")
        _mark(db, job_id, "failed",
              completed_at=datetime.now(timezone.utc),
              error_msg=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_user_from_db(db, user_pk: str, user_id: str):
    """
    Build a training-ready feature matrix from live DB data.

    Uses CgmReading (15-min resampled glucose), FoodLog (meal features),
    and WearableActivity (daily HR / steps broadcast to 15-min intervals).
    Missing wearable signals (eda, bvp, etc.) are zero-filled downstream by
    safe_get_feature inside the feature engineering pipeline.
    """
    import pandas as pd
    from src.db.models import CgmReading, FoodLog, WearableActivity
    from src.features.pipeline import build_feature_matrix

    # ── CGM ───────────────────────────────────────────────────────────────────
    cgm_rows = (
        db.query(CgmReading)
        .filter(CgmReading.user_id == user_pk)
        .order_by(CgmReading.timestamp)
        .all()
    )
    if not cgm_rows:
        raise ValueError(f"No CGM readings in DB for user {user_id}")

    cgm_df = pd.DataFrame([{
        "timestamp":    r.timestamp,
        "glucose_mg_dl": r.glucose_mgdl,
    } for r in cgm_rows])
    cgm_df["timestamp"] = pd.to_datetime(cgm_df["timestamp"], utc=True)
    cgm_df = cgm_df.set_index("timestamp").sort_index()
    cgm_df = cgm_df.resample("15min").mean()
    cgm_df["glucose_mg_dl"] = cgm_df["glucose_mg_dl"].interpolate(
        method="linear", limit=3
    )

    # ── Wearable activity (daily summaries → broadcast to 15-min rows) ────────
    activity_rows = (
        db.query(WearableActivity)
        .filter(WearableActivity.user_id_fk == user_pk)
        .order_by(WearableActivity.calendar_date)
        .all()
    )

    cgm_df["date"] = cgm_df.index.date

    if activity_rows:
        act_df = pd.DataFrame([{
            "date":             pd.to_datetime(r.calendar_date).date(),
            "hr":               r.hr_avg_bpm,
            "steps":            float(r.steps) / 96.0 if r.steps else 0.0,
            "calories_burned":  float(r.calories_active) / 96.0 if r.calories_active else 0.0,
            "mets":             float(r.calories_active) / 96.0 / 1000.0 if r.calories_active else 0.0,
        } for r in activity_rows])
        merged = cgm_df.reset_index().merge(act_df, on="date", how="left").set_index("timestamp")
    else:
        merged = cgm_df.copy()

    merged = merged.drop(columns=["date"], errors="ignore")

    # ── Food logs → meal features at matching 15-min bins ────────────────────
    food_rows = (
        db.query(FoodLog)
        .filter(FoodLog.user_id_fk == user_pk)
        .order_by(FoodLog.logged_at)
        .all()
    )

    if food_rows:
        food_df = pd.DataFrame([{
            "timestamp":          r.logged_at,
            "calorie":            float(r.calories or 0),
            "total_carb":         float(r.carbs_g   or 0),
            "protein":            float(r.protein_g  or 0),
            "total_fat":          float(r.fat_g      or 0),
            "dietary_fiber":      float(r.fiber_g    or 0),
            "sugar":              float(r.sugar_g    or 0),
            "gi_proxy":           float(r.gi_proxy   or 0),
            "amount_consumed_pct": 100.0,
            "meal_type_encoded":  _MEAL_TYPE_MAP.get((r.meal_type or "").lower(), 0),
        } for r in food_rows])
        food_df["timestamp"] = pd.to_datetime(food_df["timestamp"], utc=True)
        food_df = food_df.set_index("timestamp").sort_index()
        food_df.index = food_df.index.round("15min")

        agg = {
            "calorie": "sum", "total_carb": "sum", "protein": "sum",
            "total_fat": "sum", "dietary_fiber": "sum", "sugar": "sum",
            "gi_proxy": "mean", "amount_consumed_pct": "mean",
            "meal_type_encoded": "max",
        }
        food_df = food_df.groupby(food_df.index).agg(agg)
        merged = merged.join(food_df, how="left")

    meal_cols = ["calorie", "total_carb", "protein", "total_fat",
                 "dietary_fiber", "sugar", "gi_proxy", "amount_consumed_pct",
                 "meal_type_encoded"]
    for col in meal_cols:
        if col not in merged.columns:
            merged[col] = 0.0
        else:
            merged[col] = merged[col].fillna(0.0)

    merged = merged.fillna(0.0)
    merged = merged.dropna(subset=["glucose_mg_dl"])

    return build_feature_matrix(merged, user_id=user_id)


# ── Model zoo runner ──────────────────────────────────────────────────────────

def _run_zoo(df, dataset: str, user_id: str, horizon: str):
    """
    Train every model in the zoo on df.  Returns (best_TrainResult, best_model_name).
    Raises RuntimeError if all models fail.
    """
    from src.models.individual.trainer import IndividualTrainer
    from src.models.zoo import MODEL_REGISTRY, get_model

    best_result, best_name = None, None

    for name in MODEL_REGISTRY:
        try:
            model   = get_model(name)
            trainer = IndividualTrainer(model, dataset, user_id, horizon)
            result  = trainer.run(df)
            if best_result is None or result.val_rmse < best_result.val_rmse:
                best_result, best_name = result, name
        except Exception as exc:
            log.warning(f"Zoo model {name} failed for user {user_id}: {exc}")

    if best_result is None:
        raise RuntimeError(
            f"All zoo models failed for {dataset}/{user_id}/{horizon}."
        )
    return best_result, best_name


# ── Registry helpers ──────────────────────────────────────────────────────────

def _get_population_rmse(registry_path: Path, dataset: str, horizon: str) -> float | None:
    if not registry_path.exists():
        return None
    with open(registry_path) as f:
        reg = json.load(f)
    slot = reg.get("population", {}).get(dataset, {}).get(horizon)
    return slot["test_rmse"] if slot else None


def _get_prev_rmse(
    registry_path: Path, dataset: str, horizon: str, user_id: str
) -> float | None:
    """Current individual RMSE in registry, falling back to population."""
    if not registry_path.exists():
        return None
    with open(registry_path) as f:
        reg = json.load(f)
    ind_slot = reg.get("individual", {}).get(dataset, {}).get(user_id, {}).get(horizon)
    if ind_slot:
        return ind_slot.get("test_rmse")
    pop_slot = reg.get("population", {}).get(dataset, {}).get(horizon)
    return pop_slot["test_rmse"] if pop_slot else None


def _update_registry_individual(
    registry_path: Path,
    dataset: str,
    horizon: str,
    user_id: str,
    model_type: str,
    artefact_dir: Path,
    val_rmse: float,
    test_rmse: float,
    clarke_a_pct: float,
    version_tag: str,
) -> None:
    if registry_path.exists():
        with open(registry_path) as f:
            reg = json.load(f)
    else:
        reg = {"version": "1.0.0", "population": {}, "virtual": {}, "individual": {}}

    reg.setdefault("individual", {})
    reg["individual"].setdefault(dataset, {})
    reg["individual"][dataset].setdefault(user_id, {})

    reg["individual"][dataset][user_id][horizon] = {
        "best_model_type":     model_type,
        "val_rmse":            val_rmse,
        "test_rmse":           test_rmse,
        "clarke_a_pct":        clarke_a_pct,
        "artefact_dir":        str(artefact_dir),
        "version_tag":         version_tag,
        "trained_at":          datetime.now(timezone.utc).isoformat(),
        "min_hr_readings":     8,
        "full_hr_readings":    24,
        "hr_gap_tolerance_min": 30,
    }
    reg["updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(registry_path, "w") as f:
        json.dump(reg, f, indent=2)


def _invalidate_cache(dataset: str, horizon: str, user_id: str) -> None:
    try:
        from src.serving.model_loader import invalidate_user_cache
        invalidate_user_cache(dataset, horizon, user_id)
    except Exception as exc:
        log.warning(f"Could not invalidate model cache: {exc}")


# ── Private DB helper ─────────────────────────────────────────────────────────

def _mark(db, job_id: str, status: str, **kwargs) -> None:
    from src.db.crud import update_retrain_job
    update_retrain_job(db, job_id, status=status, **kwargs)
    db.commit()


# ── Drift-triggered retrain ───────────────────────────────────────────────────

def maybe_trigger_retrain(
    user_id_fk: str,
    user_id: str,
    dataset: str,
    horizon: str,
    current_rmse: float,
    registry_rmse: float,
) -> bool:
    """
    Call from the API after logging a prediction.
    Enqueues a retrain job if drift threshold is exceeded.
    Returns True if a job was enqueued.
    """
    from src.db.crud import create_retrain_job, get_pending_retrain_jobs, update_retrain_job
    from src.db.session import SessionLocal

    if current_rmse <= registry_rmse * (1 + DRIFT_THRESHOLD):
        return False

    db = SessionLocal()
    try:
        pending = get_pending_retrain_jobs(db, limit=MAX_RETRAIN_QUEUE)
        if len(pending) >= MAX_RETRAIN_QUEUE:
            log.warning(
                f"Retrain queue full ({MAX_RETRAIN_QUEUE}). "
                f"Skipping drift trigger for {dataset}/{horizon} user={user_id}."
            )
            return False

        job = create_retrain_job(
            db, user_id_fk, dataset, horizon,
            triggered_by="auto_drift",
            notes=(
                f"RMSE drift: live={current_rmse:.2f} > "
                f"registry={registry_rmse:.2f} × {1 + DRIFT_THRESHOLD:.2f}"
            ),
        )
        db.commit()
        db.refresh(job)

        task = trigger_retrain.delay(job.id, user_id, dataset, horizon)
        update_retrain_job(db, job.id, status="pending", celery_id=task.id)
        db.commit()
        log.info(
            f"Drift detected ({current_rmse:.2f} > {registry_rmse:.2f} + {DRIFT_THRESHOLD*100:.0f}%). "
            f"Retrain enqueued: job={job.id} celery={task.id}"
        )
        return True
    finally:
        db.close()
