from sqlalchemy.orm import Session
from typing import List, Optional
from src.backend.db.models import (
    Lecture, Classroom, ClassroomStudent, User, Attendance, BehaviorEvent, EngagementMetric, EngagementTimeline, ProcessingSession
)
from src.backend.schemas.lectures import LectureCreate, LectureUpdate, LectureSummary, LectureResponse
from src.backend.services.classroom_service import get_classroom_by_id

def get_lectures(db: Session, user: User, classroom_id: Optional[int] = None, skip: int = 0, limit: int = 100) -> List[Lecture]:
    query = db.query(Lecture)
    if user.role != "ADMIN":
        query = query.filter(Lecture.teacher_id == user.id)
    if classroom_id:
        query = query.filter(Lecture.classroom_id == classroom_id)
    return query.offset(skip).limit(limit).all()

def get_lecture_by_id(db: Session, lecture_id: int, user: User) -> Optional[Lecture]:
    query = db.query(Lecture).filter(Lecture.id == lecture_id)
    if user.role != "ADMIN":
        query = query.filter(Lecture.teacher_id == user.id)
    return query.first()

def create_lecture(db: Session, lecture_in: LectureCreate, user: User) -> Lecture:
    classroom = get_classroom_by_id(db, lecture_in.classroom_id, user)
    if not classroom:
        raise PermissionError("Classroom not found or access denied")
    
    lecture = Lecture(
        classroom_id=lecture_in.classroom_id,
        teacher_id=user.id,
        title=lecture_in.title.strip(),
        subject=lecture_in.subject.strip() if lecture_in.subject else None,
        start_time=lecture_in.start_time,
        end_time=lecture_in.end_time,
        status=lecture_in.status or "SCHEDULED"
    )
    db.add(lecture)
    db.commit()
    db.refresh(lecture)
    return lecture

def update_lecture(db: Session, lecture_id: int, lecture_in: LectureUpdate, user: User) -> Optional[Lecture]:
    lecture = get_lecture_by_id(db, lecture_id, user)
    if not lecture:
        return None
    
    update_data = lecture_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(lecture, field, value)
    
    db.commit()
    db.refresh(lecture)
    return lecture

def get_lecture_attendance(db: Session, lecture_id: int, user: User, student_id: Optional[int] = None) -> List[Attendance]:
    lecture = get_lecture_by_id(db, lecture_id, user)
    if not lecture:
        raise PermissionError("Lecture not found or access denied")
    
    query = db.query(Attendance).filter(Attendance.lecture_id == lecture_id)
    if student_id:
        query = query.filter(Attendance.student_id == student_id)
    return query.all()

def get_lecture_behavior(
    db: Session,
    lecture_id: int,
    user: User,
    event_type: Optional[str] = None,
    student_id: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> List[BehaviorEvent]:
    lecture = get_lecture_by_id(db, lecture_id, user)
    if not lecture:
        raise PermissionError("Lecture not found or access denied")
    
    session_ids = [s.id for s in lecture.processing_sessions]
    if not session_ids:
        return []
    
    query = db.query(BehaviorEvent).filter(BehaviorEvent.processing_session_id.in_(session_ids))
    if event_type:
        query = query.filter(BehaviorEvent.event_type == event_type)
    if student_id:
        query = query.filter(BehaviorEvent.student_id == student_id)
    if start_time is not None:
        query = query.filter(BehaviorEvent.start_time >= start_time)
    if end_time is not None:
        query = query.filter(BehaviorEvent.end_time <= end_time)
    
    return query.all()

def get_lecture_engagement(db: Session, lecture_id: int, user: User, student_id: Optional[int] = None) -> List[EngagementMetric]:
    lecture = get_lecture_by_id(db, lecture_id, user)
    if not lecture:
        raise PermissionError("Lecture not found or access denied")
    
    session_ids = [s.id for s in lecture.processing_sessions]
    if not session_ids:
        return []
    
    query = db.query(EngagementMetric).filter(EngagementMetric.processing_session_id.in_(session_ids))
    if student_id:
        query = query.filter(EngagementMetric.student_id == student_id)
    return query.all()

def get_lecture_engagement_timeline(db: Session, lecture_id: int, user: User) -> List[EngagementTimeline]:
    lecture = get_lecture_by_id(db, lecture_id, user)
    if not lecture:
        raise PermissionError("Lecture not found or access denied")
    
    session_ids = [s.id for s in lecture.processing_sessions]
    if not session_ids:
        return []
    
    return db.query(EngagementTimeline).filter(EngagementTimeline.processing_session_id.in_(session_ids)).order_by(EngagementTimeline.window_start).all()

def get_lecture_summary(db: Session, lecture_id: int, user: User) -> LectureSummary:
    lecture = get_lecture_by_id(db, lecture_id, user)
    if not lecture:
        raise PermissionError("Lecture not found or access denied")
    
    enrolled_students = db.query(ClassroomStudent).filter(
        ClassroomStudent.classroom_id == lecture.classroom_id,
        ClassroomStudent.is_active == True
    ).all()
    enrolled_student_ids = {e.student_id for e in enrolled_students}
    enrolled_count = len(enrolled_student_ids)
    
    attendance_records = db.query(Attendance).filter(Attendance.lecture_id == lecture_id).all()
    present_count = sum(
        1 for a in attendance_records
        if a.status == "PRESENT" and a.student_id in enrolled_student_ids
    )
    rate = round(present_count / enrolled_count, 4) if enrolled_count > 0 else 0.0

    session_ids = [s.id for s in lecture.processing_sessions]
    avg_eng = None
    behavior_events_count = 0
    if session_ids:
        metrics = db.query(EngagementMetric).filter(EngagementMetric.processing_session_id.in_(session_ids)).all()
        scores = [m.engagement_score for m in metrics if m.engagement_score is not None and m.engagement_band != "INSUFFICIENT_DATA"]
        if scores:
            avg_eng = round(sum(scores) / len(scores), 1)
        behavior_events_count = db.query(BehaviorEvent).filter(BehaviorEvent.processing_session_id.in_(session_ids)).count()

    return LectureSummary(
        lecture=LectureResponse.model_validate(lecture),
        total_enrolled=enrolled_count,
        total_present=present_count,
        attendance_rate=rate,
        average_engagement=avg_eng,
        total_behavior_events=behavior_events_count
    )
