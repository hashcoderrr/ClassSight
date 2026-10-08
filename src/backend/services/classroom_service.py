from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from src.backend.db.models import (
    Classroom, ClassroomStudent, Student, User, Attendance, BehaviorEvent, EngagementMetric, Lecture, ProcessingSession
)
from src.backend.schemas.classrooms import ClassroomCreate, ClassroomUpdate
from src.backend.schemas.attendance import AttendanceSummary
from src.backend.schemas.behavior import BehaviorSummary
from src.backend.schemas.engagement import EngagementSummary
from src.backend.services.student_service import get_student_by_student_id, get_student_by_id

def get_classrooms(db: Session, user: User, skip: int = 0, limit: int = 100) -> List[Classroom]:
    query = db.query(Classroom)
    if user.role != "ADMIN":
        query = query.filter(Classroom.teacher_id == user.id)
    return query.offset(skip).limit(limit).all()

def get_classroom_by_id(db: Session, classroom_id: int, user: User) -> Optional[Classroom]:
    query = db.query(Classroom).filter(Classroom.id == classroom_id)
    if user.role != "ADMIN":
        query = query.filter(Classroom.teacher_id == user.id)
    return query.first()

def create_classroom(db: Session, classroom_in: ClassroomCreate, teacher_id: int) -> Classroom:
    classroom = Classroom(
        name=classroom_in.name.strip(),
        room_number=classroom_in.room_number.strip() if classroom_in.room_number else None,
        description=classroom_in.description.strip() if classroom_in.description else None,
        teacher_id=teacher_id
    )
    db.add(classroom)
    db.commit()
    db.refresh(classroom)
    return classroom

def update_classroom(db: Session, classroom_id: int, classroom_in: ClassroomUpdate, user: User) -> Optional[Classroom]:
    classroom = get_classroom_by_id(db, classroom_id, user)
    if not classroom:
        return None
    
    update_data = classroom_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(classroom, field, value)
    
    db.commit()
    db.refresh(classroom)
    return classroom

def delete_classroom(db: Session, classroom_id: int, user: User) -> bool:
    classroom = get_classroom_by_id(db, classroom_id, user)
    if not classroom:
        return False
    db.delete(classroom)
    db.commit()
    return True

def get_classroom_students(db: Session, classroom_id: int, user: User) -> List[ClassroomStudent]:
    classroom = get_classroom_by_id(db, classroom_id, user)
    if not classroom:
        raise PermissionError("Classroom not found or access denied")
    return db.query(ClassroomStudent).filter(ClassroomStudent.classroom_id == classroom_id).all()

def add_student_to_classroom(db: Session, classroom_id: int, student_identifier: str, user: User) -> ClassroomStudent:
    classroom = get_classroom_by_id(db, classroom_id, user)
    if not classroom:
        raise PermissionError("Classroom not found or access denied")
    
    student = None
    if student_identifier.isdigit():
        student = get_student_by_id(db, int(student_identifier))
    if not student:
        student = get_student_by_student_id(db, student_identifier)
    
    if not student:
        raise ValueError(f"Student '{student_identifier}' not found")
    
    existing = db.query(ClassroomStudent).filter(
        ClassroomStudent.classroom_id == classroom_id,
        ClassroomStudent.student_id == student.id
    ).first()
    
    if existing:
        if not existing.is_active:
            existing.is_active = True
            db.commit()
            db.refresh(existing)
        return existing
    
    entry = ClassroomStudent(
        classroom_id=classroom_id,
        student_id=student.id,
        is_active=True
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry

def remove_student_from_classroom(db: Session, classroom_id: int, student_id: int, user: User) -> bool:
    classroom = get_classroom_by_id(db, classroom_id, user)
    if not classroom:
        raise PermissionError("Classroom not found or access denied")
    
    entry = db.query(ClassroomStudent).filter(
        ClassroomStudent.classroom_id == classroom_id,
        ClassroomStudent.student_id == student_id
    ).first()
    
    if not entry:
        return False
    
    db.delete(entry)
    db.commit()
    return True

def get_classroom_attendance_summary(db: Session, classroom_id: int, user: User) -> AttendanceSummary:
    classroom = get_classroom_by_id(db, classroom_id, user)
    if not classroom:
        raise PermissionError("Classroom not found or access denied")
    
    # Get all lectures for this classroom
    lecture_ids = [l.id for l in classroom.lectures]
    total_students = db.query(ClassroomStudent).filter(ClassroomStudent.classroom_id == classroom_id, ClassroomStudent.is_active == True).count()
    
    if not lecture_ids:
        return AttendanceSummary(
            total_students=total_students,
            present=0,
            absent=0,
            uncertain=0,
            attendance_rate=0.0
        )
    
    attendance_records = db.query(Attendance).filter(Attendance.lecture_id.in_(lecture_ids)).all()
    present = sum(1 for a in attendance_records if a.status == "PRESENT")
    absent = sum(1 for a in attendance_records if a.status == "ABSENT")
    uncertain = sum(1 for a in attendance_records if a.status == "UNCERTAIN")
    total_recs = len(attendance_records)
    rate = round(present / total_recs, 4) if total_recs > 0 else 0.0

    return AttendanceSummary(
        total_students=total_students,
        present=present,
        absent=absent,
        uncertain=uncertain,
        attendance_rate=rate
    )

def get_classroom_behavior_summary(db: Session, classroom_id: int, user: User) -> BehaviorSummary:
    classroom = get_classroom_by_id(db, classroom_id, user)
    if not classroom:
        raise PermissionError("Classroom not found or access denied")
    
    lecture_ids = [l.id for l in classroom.lectures]
    if not lecture_ids:
        return BehaviorSummary(total_events=0, event_counts={}, durations_by_type={})
    
    session_ids = [s.id for s in db.query(ProcessingSession).filter(ProcessingSession.lecture_id.in_(lecture_ids)).all()]
    if not session_ids:
        return BehaviorSummary(total_events=0, event_counts={}, durations_by_type={})
    
    events = db.query(BehaviorEvent).filter(BehaviorEvent.processing_session_id.in_(session_ids)).all()
    counts = {}
    durations = {}
    for ev in events:
        counts[ev.event_type] = counts.get(ev.event_type, 0) + 1
        durations[ev.event_type] = round(durations.get(ev.event_type, 0.0) + ev.duration, 2)
    
    return BehaviorSummary(
        total_events=len(events),
        event_counts=counts,
        durations_by_type=durations
    )

def get_classroom_engagement_summary(db: Session, classroom_id: int, user: User) -> EngagementSummary:
    classroom = get_classroom_by_id(db, classroom_id, user)
    if not classroom:
        raise PermissionError("Classroom not found or access denied")
    
    lecture_ids = [l.id for l in classroom.lectures]
    if not lecture_ids:
        return EngagementSummary(
            average_engagement=0.0,
            median_engagement=0.0,
            distribution={"HIGH": 0, "MODERATE": 0, "LOW": 0, "INSUFFICIENT_DATA": 0},
            total_scored=0,
            insufficient_data_count=0
        )
    
    session_ids = [s.id for s in db.query(ProcessingSession).filter(ProcessingSession.lecture_id.in_(lecture_ids)).all()]
    if not session_ids:
        return EngagementSummary(
            average_engagement=0.0,
            median_engagement=0.0,
            distribution={"HIGH": 0, "MODERATE": 0, "LOW": 0, "INSUFFICIENT_DATA": 0},
            total_scored=0,
            insufficient_data_count=0
        )
    
    metrics = db.query(EngagementMetric).filter(EngagementMetric.processing_session_id.in_(session_ids)).all()
    scores = [m.engagement_score for m in metrics if m.engagement_score is not None and m.engagement_band != "INSUFFICIENT_DATA"]
    dist = {"HIGH": 0, "MODERATE": 0, "LOW": 0, "INSUFFICIENT_DATA": 0}
    insufficient = 0
    for m in metrics:
        if m.engagement_band in dist:
            dist[m.engagement_band] += 1
        if m.engagement_band == "INSUFFICIENT_DATA":
            insufficient += 1
    
    avg = round(sum(scores) / len(scores), 1) if scores else 0.0
    import statistics
    med = round(statistics.median(scores), 1) if scores else 0.0

    return EngagementSummary(
        average_engagement=avg,
        median_engagement=med,
        distribution=dist,
        total_scored=len(scores),
        insufficient_data_count=insufficient
    )
