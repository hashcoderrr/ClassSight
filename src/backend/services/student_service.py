from sqlalchemy.orm import Session
from typing import List, Optional
from src.backend.db.models import Student, Attendance, BehaviorEvent, EngagementMetric
from src.backend.schemas.students import StudentCreate, StudentUpdate

def get_students(db: Session, skip: int = 0, limit: int = 100) -> List[Student]:
    return db.query(Student).offset(skip).limit(limit).all()

def get_student_by_id(db: Session, student_id: int) -> Optional[Student]:
    return db.query(Student).filter(Student.id == student_id).first()

def get_student_by_student_id(db: Session, student_id_str: str) -> Optional[Student]:
    return db.query(Student).filter(Student.student_id == student_id_str.strip()).first()

def create_student(db: Session, student_in: StudentCreate) -> Student:
    existing = get_student_by_student_id(db, student_in.student_id)
    if existing:
        raise ValueError(f"Student with ID '{student_in.student_id}' already exists")
    
    student = Student(
        student_id=student_in.student_id.strip(),
        name=student_in.name.strip(),
        email=student_in.email.strip() if student_in.email else None,
        roll_number=student_in.roll_number.strip() if student_in.roll_number else None,
        is_active=student_in.is_active
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return student

def update_student(db: Session, student_id: int, student_in: StudentUpdate) -> Optional[Student]:
    student = get_student_by_id(db, student_id)
    if not student:
        return None
    
    update_data = student_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(student, field, value)
    
    db.commit()
    db.refresh(student)
    return student

def delete_student(db: Session, student_id: int) -> bool:
    student = get_student_by_id(db, student_id)
    if not student:
        return False
    db.delete(student)
    db.commit()
    return True

def get_student_attendance(db: Session, student_id: int) -> List[Attendance]:
    return db.query(Attendance).filter(Attendance.student_id == student_id).all()

def get_student_behavior(db: Session, student_id: int) -> List[BehaviorEvent]:
    return db.query(BehaviorEvent).filter(BehaviorEvent.student_id == student_id).all()

def get_student_engagement(db: Session, student_id: int) -> List[EngagementMetric]:
    return db.query(EngagementMetric).filter(EngagementMetric.student_id == student_id).all()
