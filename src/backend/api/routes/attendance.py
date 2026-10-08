from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from src.backend.db.database import get_db
from src.backend.schemas.attendance import AttendanceResponse, AttendanceSummary
from src.backend.services.lecture_service import get_lecture_attendance
from src.backend.services.classroom_service import get_classroom_attendance_summary
from src.backend.services.student_service import get_student_attendance
from src.backend.api.dependencies import get_current_user
from src.backend.db.models import User

router = APIRouter(prefix="/api/attendance", tags=["Attendance"])

@router.get("/lectures/{lecture_id}", response_model=List[AttendanceResponse], summary="Get attendance by lecture ID")
def get_by_lecture(lecture_id: int, student_id: Optional[int] = None, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return get_lecture_attendance(db, lecture_id, current_user, student_id=student_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/classrooms/{classroom_id}/summary", response_model=AttendanceSummary, summary="Get attendance summary by classroom ID")
def get_by_classroom(classroom_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return get_classroom_attendance_summary(db, classroom_id, current_user)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/students/{student_id}", response_model=List[AttendanceResponse], summary="Get attendance by student ID")
def get_by_student(student_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return get_student_attendance(db, student_id)
