from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from src.backend.db.database import get_db
from src.backend.schemas.lectures import LectureCreate, LectureUpdate, LectureResponse, LectureSummary
from src.backend.schemas.attendance import AttendanceResponse
from src.backend.schemas.behavior import BehaviorEventResponse
from src.backend.schemas.engagement import EngagementMetricResponse, EngagementTimelineResponse
from src.backend.services.lecture_service import (
    get_lectures, get_lecture_by_id, create_lecture, update_lecture,
    get_lecture_attendance, get_lecture_behavior, get_lecture_engagement,
    get_lecture_engagement_timeline, get_lecture_summary
)
from src.backend.api.dependencies import get_current_user
from src.backend.db.models import User

router = APIRouter(prefix="/api/lectures", tags=["Lectures"])

@router.get("", response_model=List[LectureResponse], summary="List lectures for teacher (or all for admin)")
def list_lectures(
    classroom_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return get_lectures(db, current_user, classroom_id=classroom_id, skip=skip, limit=limit)

@router.post("", response_model=LectureResponse, status_code=status.HTTP_201_CREATED, summary="Create a new lecture")
def create_new_lecture(lecture_in: LectureCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return create_lecture(db, lecture_in, current_user)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

@router.get("/{id}", response_model=LectureResponse, summary="Get lecture details")
def get_lecture(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    lecture = get_lecture_by_id(db, id, current_user)
    if not lecture:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lecture not found or access denied")
    return lecture

@router.patch("/{id}", response_model=LectureResponse, summary="Update lecture status/details")
def update_lecture_details(id: int, lecture_in: LectureUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    lecture = update_lecture(db, id, lecture_in, current_user)
    if not lecture:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lecture not found or access denied")
    return lecture

@router.get("/{id}/attendance", response_model=List[AttendanceResponse], summary="Get attendance records for a lecture")
def get_attendance(id: int, student_id: Optional[int] = None, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return get_lecture_attendance(db, id, current_user, student_id=student_id)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

@router.get("/{id}/attendance/{student_id}", response_model=List[AttendanceResponse], summary="Get attendance for a specific student in a lecture")
def get_attendance_student(id: int, student_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return get_lecture_attendance(db, id, current_user, student_id=student_id)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

@router.get("/{id}/behavior", response_model=List[BehaviorEventResponse], summary="Get behavior events for a lecture with optional filters")
def get_behavior(
    id: int,
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    student_id: Optional[int] = Query(None, description="Filter by student ID"),
    start_time: Optional[float] = Query(None, description="Filter by min start_time"),
    end_time: Optional[float] = Query(None, description="Filter by max end_time"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return get_lecture_behavior(db, id, current_user, event_type=event_type, student_id=student_id, start_time=start_time, end_time=end_time)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

@router.get("/{id}/engagement", response_model=List[EngagementMetricResponse], summary="Get engagement metrics for students in a lecture")
def get_engagement(id: int, student_id: Optional[int] = None, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return get_lecture_engagement(db, id, current_user, student_id=student_id)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

@router.get("/{id}/engagement/timeline", response_model=List[EngagementTimelineResponse], summary="Get time-window engagement timeline for a lecture")
def get_timeline(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return get_lecture_engagement_timeline(db, id, current_user)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

@router.get("/{id}/summary", response_model=LectureSummary, summary="Get comprehensive lecture analytical summary")
def get_summary(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return get_lecture_summary(db, id, current_user)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
