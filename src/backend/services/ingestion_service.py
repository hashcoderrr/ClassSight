import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from sqlalchemy.orm import Session

from src.backend.core.config import settings
from src.backend.db.models import (
    Attendance,
    BehaviorEvent,
    ClassroomStudent,
    EngagementMetric,
    EngagementTimeline,
    Lecture,
    ProcessingSession,
    Report,
    Student,
    Track,
)


def _resolve_or_create_student(db: Session, student_id_str: Optional[str]) -> Optional[Student]:
    """Resolve an existing student by student_id or create a new student record."""
    if not student_id_str or str(student_id_str).startswith("track_"):
        return None
    student_id_clean = str(student_id_str).strip()
    student = db.query(Student).filter(Student.student_id == student_id_clean).first()
    if not student:
        student = Student(
            student_id=student_id_clean,
            name=f"Student {student_id_clean}",
            is_active=True,
        )
        db.add(student)
        db.flush()
    return student


def ingest_ai_report(
    db: Session,
    report_data: Dict[str, Any],
    lecture_id: Optional[int] = None,
    video_source: Optional[str] = None,
    report_file_path: Optional[str] = None,
    min_attendance_duration: Optional[float] = None,
    min_attendance_observations: Optional[int] = None,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingests Phase 3 behavior and Phase 4 engagement reports into PostgreSQL/SQLite database.

    Provides idempotency protection, atomic transactions, and robust attendance derivation.
    """
    min_duration = (
        min_attendance_duration
        if min_attendance_duration is not None
        else settings.MIN_ATTENDANCE_DURATION
    )
    min_observations = (
        min_attendance_observations
        if min_attendance_observations is not None
        else settings.MIN_ATTENDANCE_OBSERVATIONS
    )

    # 1. Deterministic Idempotency Key
    if not idempotency_key:
        fingerprint_raw = f"{lecture_id}:{json.dumps(report_data, sort_keys=True)}"
        idempotency_key = hashlib.sha256(fingerprint_raw.encode("utf-8")).hexdigest()

    existing_session = (
        db.query(ProcessingSession)
        .filter(ProcessingSession.idempotency_key == idempotency_key)
        .first()
    )
    if existing_session:
        tracks_count = (
            db.query(Track)
            .filter(Track.processing_session_id == existing_session.id)
            .count()
        )
        events_count = (
            db.query(BehaviorEvent)
            .filter(BehaviorEvent.processing_session_id == existing_session.id)
            .count()
        )
        metrics_count = (
            db.query(EngagementMetric)
            .filter(EngagementMetric.processing_session_id == existing_session.id)
            .count()
        )
        attendance_count = (
            db.query(Attendance)
            .filter(Attendance.lecture_id == lecture_id)
            .count()
            if lecture_id
            else 0
        )
        return {
            "session_id": existing_session.id,
            "lecture_id": lecture_id,
            "status": existing_session.status,
            "tracks_count": tracks_count,
            "events_count": events_count,
            "metrics_count": metrics_count,
            "attendance_count": attendance_count,
            "message": "Report already ingested (idempotent skip)",
            "is_duplicate": True,
        }

    # Extract session metadata
    session_info = report_data.get("session", {})
    video_name = (
        video_source
        or session_info.get("video")
        or report_data.get("video")
        or "unknown_video.mp4"
    )
    total_frames = (
        session_info.get("total_frames") or report_data.get("total_frames") or 0
    )
    processed_frames = report_data.get("processed_frames") or total_frames
    fps = session_info.get("fps") or report_data.get("average_fps") or 30.0

    # 2. Create ProcessingSession (within transaction)
    proc_session = ProcessingSession(
        lecture_id=lecture_id,
        idempotency_key=idempotency_key,
        video_source=video_name,
        status="COMPLETED",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        processing_fps=float(fps),
        total_frames=int(total_frames),
        processed_frames=int(processed_frames),
    )
    db.add(proc_session)
    db.flush()

    tracks_created = 0
    events_created = 0
    metrics_created = 0

    seen_students_summary = {}  # student_id -> {"visible_duration": float, "obs": int}
    tracks_by_tid = {}  # tid -> Track

    # 3. Process Students & Tracks from Phase 4 format or Phase 3 format
    students_data = report_data.get("students", {})
    if students_data:
        for key, sdata in students_data.items():
            tid = sdata.get("track_id")
            sid_str = sdata.get("student_id") or (None if key.startswith("track_") else key)
            student_obj = _resolve_or_create_student(db, sid_str) if sid_str else None
            student_db_id = student_obj.id if student_obj else None

            features = sdata.get("features", {})
            visible_dur = float(features.get("visible_duration", 0.0))

            if tid is not None:
                track = Track(
                    processing_session_id=proc_session.id,
                    track_id=int(tid),
                    student_id=student_db_id,
                    visible_duration=visible_dur,
                    identity_status="CONFIRMED" if student_db_id else "UNKNOWN",
                )
                db.add(track)
                tracks_by_tid[int(tid)] = track
                tracks_created += 1

            if student_obj:
                cur = seen_students_summary.get(
                    student_obj.id, {"visible_duration": 0.0, "obs": 0}
                )
                cur["visible_duration"] += visible_dur
                cur["obs"] += 1
                seen_students_summary[student_obj.id] = cur

            # Create EngagementMetric record
            score = sdata.get("engagement_indicator")
            metric = EngagementMetric(
                processing_session_id=proc_session.id,
                student_id=student_db_id,
                track_id=int(tid) if tid is not None else None,
                visible_duration=visible_dur,
                forward_ratio=float(features.get("forward_ratio", 0.0)),
                head_down_ratio=float(features.get("head_down_ratio", 0.0)),
                phone_ratio=float(features.get("phone_ratio", 0.0)),
                raised_hand_count=int(features.get("raised_hand_count", 0)),
                raised_hand_rate=float(features.get("raised_hand_rate", 0.0)),
                sleep_ratio=float(features.get("sleep_ratio", 0.0)),
                engagement_score=float(score) if score is not None else None,
                engagement_band=sdata.get("engagement_level", "INSUFFICIENT_DATA"),
                data_quality=sdata.get("data_quality", "LOW"),
                contributing_factors=sdata.get("contributing_factors", []),
            )
            db.add(metric)
            metrics_created += 1

    # 4. Process Behavior Events (from per_student_events or raw events)
    per_student_events = report_data.get("per_student_events", {})
    all_events = []
    if per_student_events:
        for sid_key, ev_list in per_student_events.items():
            all_events.extend(ev_list)
    elif "events" in report_data and isinstance(report_data["events"], list):
        all_events = report_data["events"]

    for ev in all_events:
        tid = ev.get("track_id", -1)
        sid_str = ev.get("student_id")
        student_obj = _resolve_or_create_student(db, sid_str) if sid_str else None
        student_db_id = student_obj.id if student_obj else None

        if tid != -1 and tid not in tracks_by_tid:
            track = Track(
                processing_session_id=proc_session.id,
                track_id=int(tid),
                student_id=student_db_id,
                visible_duration=float(ev.get("duration", 0.0)),
                identity_status="CONFIRMED" if student_db_id else "UNKNOWN",
            )
            db.add(track)
            tracks_by_tid[tid] = track
            tracks_created += 1

        b_event = BehaviorEvent(
            processing_session_id=proc_session.id,
            track_id=int(tid),
            student_id=student_db_id,
            event_type=str(ev.get("event_type", "unknown")),
            confidence=float(ev.get("confidence", 0.0))
            if ev.get("confidence") is not None
            else None,
            start_time=float(ev.get("start_time", 0.0)),
            end_time=float(ev.get("end_time", 0.0)),
            duration=float(ev.get("duration", 0.0)),
            start_frame=int(ev.get("start_frame"))
            if ev.get("start_frame") is not None
            else None,
            end_frame=int(ev.get("end_frame"))
            if ev.get("end_frame") is not None
            else None,
        )
        db.add(b_event)
        events_created += 1

    # If behavior-only report (no students dict was provided), derive seen_students_summary from events
    if not seen_students_summary and all_events:
        for ev in all_events:
            sid_str = ev.get("student_id")
            if sid_str and not str(sid_str).startswith("track_"):
                student_obj = _resolve_or_create_student(db, sid_str)
                if student_obj:
                    cur = seen_students_summary.get(
                        student_obj.id, {"visible_duration": 0.0, "obs": 0}
                    )
                    cur["visible_duration"] += float(ev.get("duration", 0.0))
                    cur["obs"] += 1
                    seen_students_summary[student_obj.id] = cur

    # 5. Process Timeline (from Phase 4 report)
    timeline_data = report_data.get("timeline", [])
    for item in timeline_data:
        tl_entry = EngagementTimeline(
            processing_session_id=proc_session.id,
            window_start=float(item.get("start", 0.0)),
            window_end=float(item.get("end", 0.0)),
            engagement_indicator=float(item["engagement_indicator"])
            if item.get("engagement_indicator") is not None
            else None,
            student_count=int(item.get("student_count", 0)),
            participation_count=0,
            phone_event_count=0,
            head_down_duration=0.0,
            sleep_event_count=0,
        )
        db.add(tl_entry)

    # 6. Process Attendance with Upsert (if lecture_id provided)
    attendance_count = 0
    if lecture_id:
        lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
        if lecture:
            # Check enrolled students in classroom
            enrolled = (
                db.query(ClassroomStudent)
                .filter(
                    ClassroomStudent.classroom_id == lecture.classroom_id,
                    ClassroomStudent.is_active == True,
                )
                .all()
            )
            enrolled_student_ids = {e.student_id for e in enrolled}

            # Upsert attendance for all observed students
            for student_db_id, stats in seen_students_summary.items():
                dur = stats["visible_duration"]
                obs = stats["obs"]
                if dur >= min_duration or obs >= min_observations:
                    status = "PRESENT"
                elif dur > 0:
                    status = "UNCERTAIN"
                else:
                    status = "ABSENT"

                confidence = 1.0 if status == "PRESENT" else 0.5

                existing_att = (
                    db.query(Attendance)
                    .filter(
                        Attendance.lecture_id == lecture.id,
                        Attendance.student_id == student_db_id,
                    )
                    .first()
                )
                if existing_att:
                    existing_att.visible_duration = dur
                    existing_att.status = status
                    existing_att.confidence = confidence
                    existing_att.last_seen_at = datetime.now(timezone.utc)
                else:
                    att = Attendance(
                        lecture_id=lecture.id,
                        student_id=student_db_id,
                        first_seen_at=datetime.now(timezone.utc),
                        last_seen_at=datetime.now(timezone.utc),
                        visible_duration=dur,
                        status=status,
                        confidence=confidence,
                    )
                    db.add(att)

                enrolled_student_ids.discard(student_db_id)

            # Record or update ABSENT for enrolled students who were not seen
            for absent_student_id in enrolled_student_ids:
                existing_att = (
                    db.query(Attendance)
                    .filter(
                        Attendance.lecture_id == lecture.id,
                        Attendance.student_id == absent_student_id,
                    )
                    .first()
                )
                if existing_att:
                    existing_att.status = "ABSENT"
                    existing_att.visible_duration = 0.0
                    existing_att.confidence = 1.0
                else:
                    att = Attendance(
                        lecture_id=lecture.id,
                        student_id=absent_student_id,
                        visible_duration=0.0,
                        status="ABSENT",
                        confidence=1.0,
                    )
                    db.add(att)

            db.flush()
            attendance_count = (
                db.query(Attendance)
                .filter(Attendance.lecture_id == lecture.id)
                .count()
            )

    # 7. Report Record Metadata
    report_type = "COMBINED"
    if "students" in report_data and "timeline" in report_data:
        report_type = "ENGAGEMENT"
    elif "per_student_events" in report_data:
        report_type = "BEHAVIOR"

    rep_record = Report(
        processing_session_id=proc_session.id,
        report_type=report_type,
        file_path=report_file_path or f"outputs/reports/{video_name}.json",
    )
    db.add(rep_record)

    db.commit()
    db.refresh(proc_session)

    return {
        "session_id": proc_session.id,
        "lecture_id": lecture_id,
        "status": "COMPLETED",
        "tracks_count": tracks_created,
        "events_count": events_created,
        "metrics_count": metrics_created,
        "attendance_count": attendance_count,
        "message": f"Successfully ingested {report_type} report into database",
        "is_duplicate": False,
    }
