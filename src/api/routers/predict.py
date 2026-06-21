"""
POST /predict          — blood glucose prediction
GET  /predict/model-info    — current model tier + metrics for authenticated user
POST /predict/request-retrain — patient self-requests personalised model training
GET  /predict/retrain-status  — patient polls their latest retrain job

HR gate runs before every prediction call. Missing HR or a gap > 30 min returns
HTTP 503 with a structured DataGapResponse — never a degraded prediction.
"""

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.deps import get_db, get_optional_user, get_current_user
from src.api.schemas import (
    DataGapResponse,
    ModelInfoResponse,
    PREDICT_OPENAPI_EXAMPLES,
    PredictRequest,
    PredictResponse,
    RetrainStatusResponse,
)
from src.db.models import User
from src.serving.inference import run_inference, validate_hr_continuity
from src.utils import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["predict"])


# ── POST /predict ─────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=PredictResponse,
    responses={503: {"model": DataGapResponse, "description": "Watch data gap — prediction blocked"}},
    summary="Run blood glucose prediction",
    response_description="Predicted glucose trajectory, model tier, Clarke zone, and risk flags",
)
def predict(
    request: PredictRequest = Body(..., openapi_examples=PREDICT_OPENAPI_EXAMPLES),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_optional_user),
) -> PredictResponse:
    # ── HR gate — first thing, before any model loading ───────────────────────
    gap = validate_hr_continuity(request.readings)
    if gap is not None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=gap.model_dump(mode="json"),
        )

    # Authenticated users get individual model lookup; anon gets population/virtual
    user_id = current_user.user_id if current_user is not None else None
    response = run_inference(request, user_id=user_id)

    if current_user is not None:
        _log_and_check_drift(response, current_user, db)

    return response


# ── GET /predict/model-info ───────────────────────────────────────────────────

@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Current model tier and metrics for the authenticated user",
)
def model_info(
    horizon: str = "2h",
    current_user: User = Depends(get_current_user),
) -> ModelInfoResponse:
    """
    Returns which model tier (individual / virtual / population) would be served
    for this user right now, along with its metrics and HR gate thresholds.
    """
    import json
    from src.serving.model_loader import REGISTRY_PATH, _resolve_slot, _read_registry

    if horizon not in ("2h", "3h"):
        raise HTTPException(status_code=400, detail="horizon must be '2h' or '3h'")

    reg      = _read_registry()
    dataset  = current_user.dataset
    user_id  = current_user.user_id

    # What tier would be served for cgm_active mode?
    _, _, tier = _resolve_slot(reg, dataset, horizon, user_id, "cgm_active")

    # Fetch the slot data for that tier
    if tier == "individual":
        slot = reg["individual"][dataset][user_id][horizon]
    elif tier == "virtual":
        slot = reg["virtual"][dataset][horizon]
    else:
        slot = reg["population"][dataset][horizon]

    # Check individual model status
    ind_slot = reg.get("individual", {}).get(dataset, {}).get(user_id, {}).get(horizon)
    pop_rmse = reg["population"][dataset][horizon]["test_rmse"]
    has_individual          = ind_slot is not None
    individual_beats_pop    = has_individual and ind_slot["test_rmse"] < pop_rmse

    return ModelInfoResponse(
        user_id=user_id,
        dataset=dataset,
        horizon=horizon,
        model_tier=tier,
        model_type=slot.get("best_model_type", slot.get("model_type", "unknown")),
        test_rmse=slot["test_rmse"],
        val_rmse=slot["val_rmse"],
        clarke_a_pct=slot["clarke_a_pct"],
        trained_at=slot.get("trained_at"),
        has_individual_model=has_individual,
        individual_beats_population=individual_beats_pop,
        min_hr_readings=slot.get("min_hr_readings", 8),
        full_hr_readings=slot.get("full_hr_readings", 24),
        hr_gap_tolerance_min=slot.get("hr_gap_tolerance_min", 30),
    )


# ── POST /predict/request-retrain ─────────────────────────────────────────────

@router.post(
    "/request-retrain",
    summary="Patient requests personalised model training",
    status_code=status.HTTP_202_ACCEPTED,
)
def request_retrain(
    horizon: str = "2h",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Enqueues a retrain job for the authenticated patient.
    Doctors manage retraining from /doctor/patients/{pk}/retrain.
    Patients can request it here — doctor still approves via the portal.
    The job is created as "pending" and processed by the Celery worker.
    """
    from src.db.crud import create_retrain_job, get_pending_retrain_jobs
    from src.tasks.retrain import trigger_retrain
    from src.db.crud import update_retrain_job

    if horizon not in ("2h", "3h"):
        raise HTTPException(status_code=400, detail="horizon must be '2h' or '3h'")

    # Prevent queue flooding — one pending job per user per horizon is enough
    pending = get_pending_retrain_jobs(db, limit=100)
    already_queued = [
        j for j in pending
        if j.user_id_fk == current_user.id and j.horizon == horizon
    ]
    if already_queued:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A retrain job for horizon={horizon} is already queued (id={already_queued[0].id}).",
        )

    job = create_retrain_job(
        db, current_user.id, current_user.dataset, horizon,
        triggered_by="patient_request",
    )
    db.commit()
    db.refresh(job)

    try:
        task = trigger_retrain.delay(job.id, current_user.user_id, current_user.dataset, horizon)
        update_retrain_job(db, job.id, status="pending", celery_id=task.id)
        db.commit()
    except Exception:
        pass  # broker down — job is in DB, worker picks it up when reconnected

    log.info(
        f"Patient {current_user.user_id} requested retrain: "
        f"dataset={current_user.dataset} horizon={horizon} job={job.id}"
    )
    return {"job_id": job.id, "status": "pending", "horizon": horizon}


# ── GET /predict/retrain-status ───────────────────────────────────────────────

@router.get(
    "/retrain-status",
    response_model=RetrainStatusResponse,
    summary="Latest retrain job status for the authenticated patient",
)
def retrain_status(
    horizon: str = "2h",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RetrainStatusResponse:
    from src.db.models import RetrainJob
    from sqlalchemy import desc

    if horizon not in ("2h", "3h"):
        raise HTTPException(status_code=400, detail="horizon must be '2h' or '3h'")

    job = (
        db.query(RetrainJob)
        .filter(
            RetrainJob.user_id_fk == current_user.id,
            RetrainJob.horizon == horizon,
        )
        .order_by(desc(RetrainJob.triggered_at))
        .first()
    )

    if job is None:
        return RetrainStatusResponse(
            has_active_job=False,
            job_id=None, status=None, triggered_by=None,
            triggered_at=None, completed_at=None,
            test_rmse=None, prev_rmse=None, improved=None,
            version_tag=None, error_msg=None,
        )

    return RetrainStatusResponse(
        has_active_job=job.status in ("pending", "running"),
        job_id=job.id,
        status=job.status,
        triggered_by=job.triggered_by,
        triggered_at=job.triggered_at,
        completed_at=job.completed_at,
        test_rmse=job.test_rmse,
        prev_rmse=job.prev_rmse,
        improved=job.improved,
        version_tag=job.version_tag,
        error_msg=job.error_msg,
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _log_and_check_drift(
    response: PredictResponse,
    user: User,
    db: Session,
) -> None:
    try:
        from src.db.crud import log_prediction, recent_prediction_rmse
        from src.serving.model_loader import REGISTRY_PATH
        import json

        log_prediction(
            db,
            user_id_fk=user.id,
            dataset=response.dataset,
            horizon=response.horizon,
            model_type=response.model_type,
            current_glucose=response.current_glucose,
            horizon_glucose=response.horizon_glucose,
            clarke_zone=response.clarke_zone,
            is_hypo_risk=response.is_hypo_risk,
            is_hyper_risk=response.is_hyper_risk,
            n_readings=response.n_readings_used,
            model_tier=response.model_tier,
            cgm_mode=response.cgm_mode,
            data_quality=response.data_quality,
            readings_used=response.readings_used,
        )
        db.commit()

        from src.db.models import PredictionLog
        n_logs = (
            db.query(PredictionLog)
            .filter(PredictionLog.user_id_fk == user.id)
            .count()
        )
        if n_logs % 10 != 0:
            return

        live_rmse = recent_prediction_rmse(db, user.id, response.horizon, last_n=50)
        if live_rmse is None:
            return

        if REGISTRY_PATH.exists():
            with open(REGISTRY_PATH) as f:
                reg = json.load(f)
            registry_rmse = (
                reg.get("population", {})
                .get(response.dataset, {})
                .get(response.horizon, {})
                .get("test_rmse")
            )
            if registry_rmse:
                from src.tasks.retrain import maybe_trigger_retrain
                maybe_trigger_retrain(
                    user_id_fk=user.id,
                    user_id=user.user_id,
                    dataset=response.dataset,
                    horizon=response.horizon,
                    current_rmse=live_rmse,
                    registry_rmse=registry_rmse,
                )
    except Exception as exc:
        log.warning(f"Prediction logging failed (non-fatal): {exc}")
