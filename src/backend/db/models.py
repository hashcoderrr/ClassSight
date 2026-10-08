from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, ForeignKey, Float, JSON, UniqueConstraint, Text
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from src.backend.db.database import Base

def utc_now():
    return datetime.now(timezone.utc)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), default="TEACHER", nullable=False)  # "ADMIN", "TEACHER"
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    classrooms = relationship("Classroom", back_populates="teacher", cascade="all, delete-orphan")
    lectures = relationship("Lecture", back_populates="teacher")


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(100), unique=True, index=True, nullable=False)  # e.g., "STU001"
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True)
    roll_number = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    classroom_associations = relationship("ClassroomStudent", back_populates="student", cascade="all, delete-orphan")
    attendance_records = relationship("Attendance", back_populates="student")
    tracks = relationship("Track", back_populates="student")
    behavior_events = relationship("BehaviorEvent", back_populates="student")
    engagement_metrics = relationship("EngagementMetric", back_populates="student")


class Classroom(Base):
    __tablename__ = "classrooms"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    room_number = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    teacher = relationship("User", back_populates="classrooms")
    student_associations = relationship("ClassroomStudent", back_populates="classroom", cascade="all, delete-orphan")
    lectures = relationship("Lecture", back_populates="classroom", cascade="all, delete-orphan")


class ClassroomStudent(Base):
    __tablename__ = "classroom_students"

    id = Column(Integer, primary_key=True, index=True)
    classroom_id = Column(Integer, ForeignKey("classrooms.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    enrolled_at = Column(DateTime(timezone=True), default=utc_now)
    is_active = Column(Boolean, default=True, nullable=False)

    __table_args__ = (
        UniqueConstraint("classroom_id", "student_id", name="uq_classroom_student"),
    )

    classroom = relationship("Classroom", back_populates="student_associations")
    student = relationship("Student", back_populates="classroom_associations")


class Lecture(Base):
    __tablename__ = "lectures"

    id = Column(Integer, primary_key=True, index=True)
    classroom_id = Column(Integer, ForeignKey("classrooms.id"), nullable=False)
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=False)
    subject = Column(String(255), nullable=True)
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(50), default="SCHEDULED", nullable=False)  # SCHEDULED, RUNNING, COMPLETED, CANCELLED
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    classroom = relationship("Classroom", back_populates="lectures")
    teacher = relationship("User", back_populates="lectures")
    processing_sessions = relationship("ProcessingSession", back_populates="lecture", cascade="all, delete-orphan")
    attendance_records = relationship("Attendance", back_populates="lecture", cascade="all, delete-orphan")


class ProcessingSession(Base):
    __tablename__ = "processing_sessions"

    id = Column(Integer, primary_key=True, index=True)
    lecture_id = Column(Integer, ForeignKey("lectures.id"), nullable=True)
    idempotency_key = Column(String(64), unique=True, nullable=True, index=True)
    video_source = Column(String(500), nullable=True)
    status = Column(String(50), default="COMPLETED", nullable=False)  # QUEUED, PROCESSING, COMPLETED, FAILED
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    processing_fps = Column(Float, nullable=True)
    total_frames = Column(Integer, nullable=True)
    processed_frames = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    lecture = relationship("Lecture", back_populates="processing_sessions")
    tracks = relationship("Track", back_populates="session", cascade="all, delete-orphan")
    behavior_events = relationship("BehaviorEvent", back_populates="session", cascade="all, delete-orphan")
    engagement_metrics = relationship("EngagementMetric", back_populates="session", cascade="all, delete-orphan")
    engagement_timeline = relationship("EngagementTimeline", back_populates="session", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="session", cascade="all, delete-orphan")


class Track(Base):
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, index=True)
    processing_session_id = Column(Integer, ForeignKey("processing_sessions.id"), nullable=False)
    track_id = Column(Integer, nullable=False, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=True)
    first_seen = Column(Float, nullable=True)
    last_seen = Column(Float, nullable=True)
    visible_duration = Column(Float, default=0.0)
    identity_status = Column(String(50), default="UNKNOWN")  # CONFIRMED, UNKNOWN, PENDING

    session = relationship("ProcessingSession", back_populates="tracks")
    student = relationship("Student", back_populates="tracks")


class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    lecture_id = Column(Integer, ForeignKey("lectures.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    first_seen_at = Column(DateTime(timezone=True), nullable=True)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    visible_duration = Column(Float, default=0.0)
    status = Column(String(50), default="PRESENT", nullable=False)  # PRESENT, ABSENT, UNCERTAIN
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    __table_args__ = (
        UniqueConstraint("lecture_id", "student_id", name="uq_attendance_lecture_student"),
    )

    lecture = relationship("Lecture", back_populates="attendance_records")
    student = relationship("Student", back_populates="attendance_records")


class BehaviorEvent(Base):
    __tablename__ = "behavior_events"

    id = Column(Integer, primary_key=True, index=True)
    processing_session_id = Column(Integer, ForeignKey("processing_sessions.id"), nullable=False)
    track_id = Column(Integer, nullable=False, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=True)
    event_type = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, nullable=True)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    start_frame = Column(Integer, nullable=True)
    end_frame = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    session = relationship("ProcessingSession", back_populates="behavior_events")
    student = relationship("Student", back_populates="behavior_events")


class EngagementMetric(Base):
    __tablename__ = "engagement_metrics"

    id = Column(Integer, primary_key=True, index=True)
    processing_session_id = Column(Integer, ForeignKey("processing_sessions.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=True)
    track_id = Column(Integer, nullable=True)
    visible_duration = Column(Float, default=0.0)
    forward_ratio = Column(Float, default=0.0)
    head_down_ratio = Column(Float, default=0.0)
    phone_ratio = Column(Float, default=0.0)
    raised_hand_count = Column(Integer, default=0)
    raised_hand_rate = Column(Float, default=0.0)
    sleep_ratio = Column(Float, default=0.0)
    engagement_score = Column(Float, nullable=True)
    engagement_band = Column(String(50), default="INSUFFICIENT_DATA")  # HIGH, MODERATE, LOW, INSUFFICIENT_DATA
    data_quality = Column(String(50), default="LOW")  # HIGH, MEDIUM, LOW, INSUFFICIENT_DATA
    contributing_factors = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    session = relationship("ProcessingSession", back_populates="engagement_metrics")
    student = relationship("Student", back_populates="engagement_metrics")


class EngagementTimeline(Base):
    __tablename__ = "engagement_timeline"

    id = Column(Integer, primary_key=True, index=True)
    processing_session_id = Column(Integer, ForeignKey("processing_sessions.id"), nullable=False)
    window_start = Column(Float, nullable=False)
    window_end = Column(Float, nullable=False)
    engagement_indicator = Column(Float, nullable=True)
    student_count = Column(Integer, default=0)
    participation_count = Column(Integer, default=0)
    phone_event_count = Column(Integer, default=0)
    head_down_duration = Column(Float, default=0.0)
    sleep_event_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    session = relationship("ProcessingSession", back_populates="engagement_timeline")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    processing_session_id = Column(Integer, ForeignKey("processing_sessions.id"), nullable=False)
    report_type = Column(String(100), nullable=False)  # BEHAVIOR, ENGAGEMENT, IDENTITY, ATTENDANCE, CLASSROOM_SUMMARY
    file_path = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    session = relationship("ProcessingSession", back_populates="reports")
