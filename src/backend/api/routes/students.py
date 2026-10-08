from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from src.backend.db.database import get_db
from src.backend.schemas.students import StudentCreate, StudentUpdate, StudentResponse
from src.backend.schemas.attendance import AttendanceResponse
from src.backend.schemas.behavior import BehaviorEventResponse
from src.backend.schemas.engagement import EngagementMetricResponse
from src.backend.services.student_service import (
    get_students, get_student_by_id, create_student, update_student,
    delete_student, get_student_attendance, get_student_behavior, get_student_engagement
)
from src.backend.api.dependencies import get_current_user
from src.backend.db.models import User

router = APIRouter(prefix="/api/students", tags=["Students"])

@router.get("", response_model=List[StudentResponse], summary="List enrolled students")
def list_students(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return get_students(db, skip=skip, limit=limit)

@router.post("", response_model=StudentResponse, status_code=status.HTTP_201_CREATED, summary="Create a new student")
def create_new_student(student_in: StudentCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        return create_student(db, student_in)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.get("/{id}", response_model=StudentResponse, summary="Get student details by ID")
def get_student(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    student = get_student_by_id(db, id)
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return student

@router.patch("/{id}", response_model=StudentResponse, summary="Update student details")
def update_student_details(id: int, student_in: StudentUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    student = update_student(db, id, student_in)
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return student

@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete student record")
def remove_student(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    deleted = delete_student(db, id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return None

@router.get("/{id}/attendance", response_model=List[AttendanceResponse], summary="Get attendance history for a student")
def get_attendance_for_student(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    student = get_student_by_id(db, id)
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return get_student_attendance(db, id)

@router.get("/{id}/behavior", response_model=List[BehaviorEventResponse], summary="Get behavior events for a student")
def get_behavior_for_student(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    student = get_student_by_id(db, id)
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return get_student_behavior(db, id)

@router.get("/{id}/engagement", response_model=List[EngagementMetricResponse], summary="Get engagement metrics for a student")
def get_engagement_for_student(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    student = get_student_by_id(db, id)
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return get_student_engagement(db, id)
