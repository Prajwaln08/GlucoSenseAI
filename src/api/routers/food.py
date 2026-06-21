"""
Food logging endpoints.

    POST /food/log         — log a meal entry
    GET  /food/logs        — list current user's recent food logs
    DELETE /food/logs/{id} — delete a food log entry
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.deps import get_current_user, get_db
from src.db import crud
from src.db.models import User

router = APIRouter(prefix="/food", tags=["food"])

VALID_MEAL_TYPES   = {"breakfast", "lunch", "dinner", "snack", "other"}
VALID_PORTION_SIZES = {"cup", "bowl", "katori", "glass", "plate", "piece", "serving"}


class FoodLogRequest(BaseModel):
    meal_type: str = Field(..., description="breakfast | lunch | dinner | snack | other")
    description: Optional[str] = Field(None, description="What was eaten (free text)")
    calories: Optional[float] = Field(None, description="Total calories (kcal)")
    carbs_g: Optional[float] = Field(None, description="Carbohydrates (g)")
    protein_g: Optional[float] = Field(None, description="Protein (g)")
    fat_g: Optional[float] = Field(None, description="Fat (g)")
    fiber_g: Optional[float] = Field(None, description="Dietary fiber (g)")
    sugar_g: Optional[float] = Field(None, description="Sugar (g)")
    gi_proxy: Optional[float] = Field(None, description="Glycemic index proxy (0–1)")
    quantity: Optional[float] = Field(None, description="Amount consumed (e.g. 1.5)")
    portion_size: Optional[str] = Field(None, description="cup | bowl | katori | glass | plate | piece | serving")
    notes: Optional[str] = Field(None, description="Extra notes (insulin dose, glucose reading at meal time, etc.)")
    logged_at: Optional[datetime] = Field(None, description="ISO timestamp — omit to use server time now()")

    model_config = {
        "json_schema_extra": {
            "example": {
                "meal_type": "breakfast",
                "description": "Oats with banana and almond milk",
                "calories": 380.0,
                "carbs_g": 62.0,
                "protein_g": 10.0,
                "fat_g": 8.0,
                "fiber_g": 6.0,
                "sugar_g": 14.0,
                "gi_proxy": 0.55,
                "quantity": 1.0,
                "portion_size": "bowl",
                "notes": "Pre-workout meal",
            }
        }
    }


class FoodLogResponse(BaseModel):
    id: str
    meal_type: str
    description: Optional[str]
    calories: Optional[float]
    carbs_g: Optional[float]
    protein_g: Optional[float]
    fat_g: Optional[float]
    fiber_g: Optional[float]
    sugar_g: Optional[float]
    gi_proxy: Optional[float]
    quantity: Optional[float]
    portion_size: Optional[str]
    notes: Optional[str]
    logged_at: datetime

    model_config = {"from_attributes": True}


@router.post("/log", response_model=FoodLogResponse, status_code=201)
def log_food(
    req: FoodLogRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if req.meal_type not in VALID_MEAL_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"meal_type must be one of {sorted(VALID_MEAL_TYPES)}",
        )
    if req.portion_size and req.portion_size not in VALID_PORTION_SIZES:
        raise HTTPException(
            status_code=422,
            detail=f"portion_size must be one of {sorted(VALID_PORTION_SIZES)}",
        )
    entry = crud.create_food_log(
        db,
        user_id_fk=current_user.id,
        meal_type=req.meal_type,
        description=req.description,
        calories=req.calories,
        carbs_g=req.carbs_g,
        protein_g=req.protein_g,
        fat_g=req.fat_g,
        fiber_g=req.fiber_g,
        sugar_g=req.sugar_g,
        gi_proxy=req.gi_proxy,
        quantity=req.quantity,
        portion_size=req.portion_size,
        notes=req.notes,
        logged_at=req.logged_at,
    )
    db.commit()
    db.refresh(entry)
    return entry


@router.get("/logs", response_model=list[FoodLogResponse])
def list_food_logs(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return crud.get_food_logs(db, user_id_fk=current_user.id, limit=min(limit, 200))


@router.delete("/logs/{log_id}")
def delete_food_log(
    log_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    from src.db.models import FoodLog
    entry = db.query(FoodLog).filter(
        FoodLog.id == log_id,
        FoodLog.user_id_fk == current_user.id,
    ).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Food log not found.")
    db.delete(entry)
    db.commit()
    return Response(status_code=204)
