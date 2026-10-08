from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from src.backend.db.database import get_db
from src.backend.schemas.classrooms import (
    ClassroomCreate, ClassroomUpdate, ClassroomResponse, ClassroomStudentAdd, ClassroomStudentResponse
)
from src.backend.schemas.attendance import AttendanceSummary
from src.backend.schemas.behavior import BehaviorSummary
from src.backend.schemas.engagement import EngagementSummary
from src.backend.services.classroom_service import (
    get_classrooms, get_classroom_by_id, create_classroom, update_classroom,
    delete_classroom, get_classroom_students, add_student_to_classroom,
    remove_student_from_classroom, get_classroom_attendance_summary,
    get_classroom_behavior_summary, get_classroom_engagement_summary
)
from src.backend.api.dependencies import get_current_user
from src.backend.db.models import User

router = APIRouter(prefix="/api/classrooms", tags=["Classrooms"])

@router.get("", response_model=List[ClassroomResponse], summary="List classrooms owned by teacher (or all for admin)")
def list_classrooms(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return get_classrooms(db, current_user, skip=skip, limit=limit)

@router.post("", response_model=ClassroomResponse, status_code=status.HTTP_201_CREATED, summary="Create a new classroom")
def create_new_classroom(classroom_in: ClassroomCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return create_classroom(db, classroom_in, teacher_id=current_user.id)

@router.get("/{id}", response_model=ClassroomResponse, summary="Get classroom details")
def get_classroom(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    classroom = get_classroom_by_id(db, id, current_user)
    if not classroom:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Classroom not found or access denied")
    return classroom

@router.patch("/{id}", response_model=ClassroomResponse, summary="Update classroom details")
def update_classroom_details(id: int, classroom_in: ClassroomUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    classroom = update_classroom(db, id, classroom_in, current_user)
    if not classroom:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Classroom not found or access denied")
    return classroom

@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete classroom")
def remove_classroom(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    deleted = delete_classroom(db, id, current_user)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Classroom not found or access denied")
    return None

@router.get("/{id}/students", response_model=List[ClassroomStudentResponse], summary="List students enrolled in classroom")
def list_classroom_students(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return get_classroom_students(db, id, current_user)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

@router.post("/{id}/students", response_model=ClassroomStudentResponse, status_code=status.HTTP_201_CREATED, summary="Enroll a student in classroom")
def enroll_student(id: int, student_in: ClassroomStudentAdd, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return add_student_to_classroom(db, id, student_in.student_id, current_user)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@router.delete("/{id}/students/{student_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Unenroll student from classroom")
def unenroll_student(id: int, student_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        removed = remove_student_from_classroom(db, id, student_id, current_user)
        if not removed:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Enrollment not found")
        return None
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

@router.get("/{id}/attendance/summary", response_model=AttendanceSummary, summary="Get aggregate attendance summary for classroom")
def get_attendance_summary(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return get_classroom_attendance_summary(db, id, current_user)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

@router.get("/{id}/behavior/summary", response_model=BehaviorSummary, summary="Get aggregate behavior summary for classroom")
def get_behavior_summary(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return get_classroom_behavior_summary(db, id, current_user)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

@router.get("/{id}/engagement/summary", response_model=EngagementSummary, summary="Get aggregate engagement summary for classroom")
def get_engagement_summary(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return get_classroom_engagement_summary(db, id, current_user)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
