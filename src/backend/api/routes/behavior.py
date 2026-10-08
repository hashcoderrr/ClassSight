from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from src.backend.db.database import get_db
from src.backend.schemas.behavior import BehaviorEventResponse, BehaviorSummary
from src.backend.services.lecture_service import get_lecture_behavior
from src.backend.services.classroom_service import get_classroom_behavior_summary
from src.backend.services.student_service import get_student_behavior
from src.backend.api.dependencies import get_current_user
from src.backend.db.models import User

router = APIRouter(prefix="/api/behavior", tags=["Behavior"])

@router.get("/lectures/{lecture_id}", response_model=List[BehaviorEventResponse], summary="Get behavior events by lecture ID")
def get_by_lecture(
    lecture_id: int,
    event_type: Optional[str] = Query(None),
    student_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return get_lecture_behavior(db, lecture_id, current_user, event_type=event_type, student_id=student_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/classrooms/{classroom_id}/summary", response_model=BehaviorSummary, summary="Get behavior summary by classroom ID")
def get_by_classroom(classroom_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return get_classroom_behavior_summary(db, classroom_id, current_user)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/students/{student_id}", response_model=List[BehaviorEventResponse], summary="Get behavior events by student ID")
def get_by_student(student_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return get_student_behavior(db, student_id)
