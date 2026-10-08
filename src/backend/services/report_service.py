import csv
import io
import json
from typing import Any, Dict, List, Tuple
from sqlalchemy.orm import Session

from src.backend.db.models import User
from src.backend.services.lecture_service import (
    get_lecture_by_id,
    get_lecture_attendance,
    get_lecture_behavior,
    get_lecture_engagement,
    get_lecture_summary,
)


def _sanitize_csv_cell(val: Any) -> Any:
    """Sanitize CSV cell against formula injection and format None as 'N/A'."""
    if val is None:
        return "N/A"
    if isinstance(val, str) and len(val) > 0 and val[0] in ("=", "+", "-", "@", "\t", "\r"):
        return f"'{val}"
    return val


def _sanitize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Apply sanitization across all row fields."""
    return {k: _sanitize_csv_cell(v) for k, v in row.items()}


def export_behavior_report(
    db: Session, lecture_id: int, user: User, format: str = "json"
) -> Tuple[Any, str]:
    lecture = get_lecture_by_id(db, lecture_id, user)
    if not lecture:
        raise PermissionError("Lecture not found or access denied")

    events = get_lecture_behavior(db, lecture_id, user)
    event_list = [
        {
            "id": ev.id,
            "track_id": ev.track_id,
            "student_id": ev.student.student_id if ev.student else None,
            "student_name": ev.student.name if ev.student else None,
            "event_type": ev.event_type,
            "confidence": ev.confidence,
            "start_time": ev.start_time,
            "end_time": ev.end_time,
            "duration": ev.duration,
            "start_frame": ev.start_frame,
            "end_frame": ev.end_frame,
            "created_at": ev.created_at.isoformat() if ev.created_at else None,
        }
        for ev in events
    ]

    if format.lower() == "csv":
        output = io.StringIO()
        fieldnames = [
            "id",
            "track_id",
            "student_id",
            "student_name",
            "event_type",
            "confidence",
            "start_time",
            "end_time",
            "duration",
            "start_frame",
            "end_frame",
            "created_at",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for ev in event_list:
            writer.writerow(_sanitize_row(ev))
        return output.getvalue(), "text/csv"

    return {
        "lecture_id": lecture_id,
        "lecture_title": lecture.title,
        "events": event_list,
    }, "application/json"


def export_engagement_report(
    db: Session, lecture_id: int, user: User, format: str = "json"
) -> Tuple[Any, str]:
    lecture = get_lecture_by_id(db, lecture_id, user)
    if not lecture:
        raise PermissionError("Lecture not found or access denied")

    metrics = get_lecture_engagement(db, lecture_id, user)
    metric_list = [
        {
            "id": m.id,
            "student_id": m.student.student_id if m.student else None,
            "student_name": m.student.name if m.student else None,
            "track_id": m.track_id,
            "visible_duration": m.visible_duration,
            "forward_ratio": m.forward_ratio,
            "head_down_ratio": m.head_down_ratio,
            "phone_ratio": m.phone_ratio,
            "raised_hand_count": m.raised_hand_count,
            "raised_hand_rate": m.raised_hand_rate,
            "sleep_ratio": m.sleep_ratio,
            "engagement_score": m.engagement_score,
            "engagement_band": m.engagement_band,
            "data_quality": m.data_quality,
            "contributing_factors": (
                json.dumps(m.contributing_factors)
                if isinstance(m.contributing_factors, (list, dict))
                else str(m.contributing_factors)
            ),
        }
        for m in metrics
    ]

    if format.lower() == "csv":
        output = io.StringIO()
        fieldnames = [
            "id",
            "student_id",
            "student_name",
            "track_id",
            "visible_duration",
            "forward_ratio",
            "head_down_ratio",
            "phone_ratio",
            "raised_hand_count",
            "raised_hand_rate",
            "sleep_ratio",
            "engagement_score",
            "engagement_band",
            "data_quality",
            "contributing_factors",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for m in metric_list:
            writer.writerow(_sanitize_row(m))
        return output.getvalue(), "text/csv"

    return {
        "lecture_id": lecture_id,
        "lecture_title": lecture.title,
        "metrics": metric_list,
    }, "application/json"


def export_attendance_report(
    db: Session, lecture_id: int, user: User, format: str = "json"
) -> Tuple[Any, str]:
    lecture = get_lecture_by_id(db, lecture_id, user)
    if not lecture:
        raise PermissionError("Lecture not found or access denied")

    records = get_lecture_attendance(db, lecture_id, user)
    att_list = [
        {
            "id": a.id,
            "student_id": a.student.student_id if a.student else None,
            "student_name": a.student.name if a.student else None,
            "roll_number": a.student.roll_number if a.student else None,
            "status": a.status,
            "visible_duration": a.visible_duration,
            "confidence": a.confidence,
            "first_seen_at": a.first_seen_at.isoformat() if a.first_seen_at else None,
            "last_seen_at": a.last_seen_at.isoformat() if a.last_seen_at else None,
        }
        for a in records
    ]

    if format.lower() == "csv":
        output = io.StringIO()
        fieldnames = [
            "id",
            "student_id",
            "student_name",
            "roll_number",
            "status",
            "visible_duration",
            "confidence",
            "first_seen_at",
            "last_seen_at",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for a in att_list:
            writer.writerow(_sanitize_row(a))
        return output.getvalue(), "text/csv"

    return {
        "lecture_id": lecture_id,
        "lecture_title": lecture.title,
        "attendance": att_list,
    }, "application/json"


def export_summary_report(
    db: Session, lecture_id: int, user: User, format: str = "json"
) -> Tuple[Any, str]:
    summary = get_lecture_summary(db, lecture_id, user)
    data = summary.model_dump()
    if format.lower() == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Lecture ID", _sanitize_csv_cell(data["lecture"]["id"])])
        writer.writerow(["Title", _sanitize_csv_cell(data["lecture"]["title"])])
        writer.writerow(["Subject", _sanitize_csv_cell(data["lecture"]["subject"])])
        writer.writerow(["Total Enrolled", _sanitize_csv_cell(data["total_enrolled"])])
        writer.writerow(["Total Present", _sanitize_csv_cell(data["total_present"])])
        writer.writerow(
            ["Attendance Rate", f"{data['attendance_rate']*100:.1f}%"]
        )
        writer.writerow(
            [
                "Average Engagement",
                (
                    f"{data['average_engagement']:.1f}"
                    if data["average_engagement"] is not None
                    else "N/A"
                ),
            ]
        )
        writer.writerow(
            [
                "Total Behavior Events",
                _sanitize_csv_cell(data["total_behavior_events"]),
            ]
        )
        return output.getvalue(), "text/csv"

    return data, "application/json"
