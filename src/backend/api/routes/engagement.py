from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from src.backend.db.database import get_db
from src.backend.schemas.engagement import EngagementMetricResponse, EngagementTimelineResponse, EngagementSummary
from src.backend.services.lecture_service import get_lecture_engagement, get_lecture_engagement_timeline
from src.backend.services.classroom_service import get_classroom_engagement_summary
from src.backend.services.student_service import get_student_engagement
from src.backend.api.dependencies import get_current_user
from src.backend.db.models import User

router = APIRouter(prefix="/api/engagement", tags=["Engagement"])

@router.get("/lectures/{lecture_id}", response_model=List[EngagementMetricResponse], summary="Get engagement metrics by lecture ID")
def get_by_lecture(lecture_id: int, student_id: Optional[int] = None, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return get_lecture_engagement(db, lecture_id, current_user, student_id=student_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/lectures/{lecture_id}/timeline", response_model=List[EngagementTimelineResponse], summary="Get engagement timeline by lecture ID")
def get_timeline_by_lecture(lecture_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return get_lecture_engagement_timeline(db, lecture_id, current_user)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/classrooms/{classroom_id}/summary", response_model=EngagementSummary, summary="Get engagement summary by classroom ID")
def get_by_classroom(classroom_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return get_classroom_engagement_summary(db, classroom_id, current_user)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/students/{student_id}", response_model=List[EngagementMetricResponse], summary="Get engagement metrics by student ID")
def get_by_student(student_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return get_student_engagement(db, student_id)
