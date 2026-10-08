import json

def test_ingest_phase3_behavior_report(client, teacher_auth_headers):
    # Setup classroom and lecture
    c_res = client.post("/api/classrooms", json={"name": "Ingest Classroom"}, headers=teacher_auth_headers)
    cid = c_res.json()["id"]

    # Pre-enroll STU001
    st_res = client.post("/api/students", json={"student_id": "STU001", "name": "Enrolled Student 1"}, headers=teacher_auth_headers)
    sid_int = st_res.json()["id"]
    client.post(f"/api/classrooms/{cid}/students", json={"student_id": "STU001"}, headers=teacher_auth_headers)

    # Pre-enroll STU002 who will be absent
    st_res2 = client.post("/api/students", json={"student_id": "STU002", "name": "Enrolled Student 2 (Absent)"}, headers=teacher_auth_headers)
    client.post(f"/api/classrooms/{cid}/students", json={"student_id": "STU002"}, headers=teacher_auth_headers)

    l_res = client.post("/api/lectures", json={"classroom_id": cid, "title": "AI Ingestion Lecture"}, headers=teacher_auth_headers)
    lid = l_res.json()["id"]

    # Mock Phase 3 behavior report
    behavior_report = {
        "video": "classroom_lecture.mp4",
        "detection_model": "yolo11n.pt",
        "pose_model": "yolo11n-pose.pt",
        "resolution": "1920x1080",
        "total_frames": 500,
        "processed_frames": 500,
        "average_fps": 25.0,
        "unique_tracks": 2,
        "identified_students": 1,
        "unknown_tracks": 1,
        "total_behavior_events": 2,
        "per_student_events": {
            "STU001": [
                {
                    "track_id": 1,
                    "student_id": "STU001",
                    "event_type": "raised_hand",
                    "confidence": 0.95,
                    "start_time": 2.0,
                    "end_time": 6.5,
                    "duration": 4.5,
                    "start_frame": 50,
                    "end_frame": 162
                }
            ],
            "track_5": [
                {
                    "track_id": 5,
                    "student_id": None,
                    "event_type": "phone_usage",
                    "confidence": 0.88,
                    "start_time": 10.0,
                    "end_time": 15.0,
                    "duration": 5.0,
                    "start_frame": 250,
                    "end_frame": 375
                }
            ]
        }
    }

    payload = {
        "lecture_id": lid,
        "report_data": behavior_report,
        "video_source": "classroom_lecture.mp4",
        "report_file_path": "outputs/reports/behavior_report_test.json"
    }

    ingest_res = client.post("/api/processing/sessions/ingest", json=payload, headers=teacher_auth_headers)
    assert ingest_res.status_code == 201
    data = ingest_res.json()
    assert data["tracks_count"] >= 2
    assert data["events_count"] == 2
    assert data["lecture_id"] == lid

    # Verify Behavior endpoint returns events
    bev_res = client.get(f"/api/lectures/{lid}/behavior", headers=teacher_auth_headers)
    assert bev_res.status_code == 200
    events = bev_res.json()
    assert len(events) == 2
    event_types = {e["event_type"] for e in events}
    assert "raised_hand" in event_types
    assert "phone_usage" in event_types

    # Test filtering by event_type
    filtered_res = client.get(f"/api/lectures/{lid}/behavior?event_type=raised_hand", headers=teacher_auth_headers)
    assert filtered_res.status_code == 200
    assert len(filtered_res.json()) == 1
    assert filtered_res.json()[0]["event_type"] == "raised_hand"


def test_ingest_phase4_engagement_report_and_attendance(client, teacher_auth_headers):
    c_res = client.post("/api/classrooms", json={"name": "Engagement Class"}, headers=teacher_auth_headers)
    cid = c_res.json()["id"]

    # Pre-enroll students
    client.post("/api/students", json={"student_id": "STU101", "name": "Student 101"}, headers=teacher_auth_headers)
    client.post("/api/students", json={"student_id": "STU102", "name": "Student 102"}, headers=teacher_auth_headers)
    client.post(f"/api/classrooms/{cid}/students", json={"student_id": "STU101"}, headers=teacher_auth_headers)
    client.post(f"/api/classrooms/{cid}/students", json={"student_id": "STU102"}, headers=teacher_auth_headers)

    l_res = client.post("/api/lectures", json={"classroom_id": cid, "title": "Engagement Ingestion Lecture"}, headers=teacher_auth_headers)
    lid = l_res.json()["id"]

    engagement_report = {
        "session": {
            "video": "lecture_engagement.mp4",
            "duration": 60.0,
            "total_frames": 1800,
            "fps": 30.0,
            "resolution": "1920x1080"
        },
        "classroom": {
            "students_observed": 2,
            "identified_students": 1,
            "unknown_tracks": 1,
            "scored_students": 1,
            "average_engagement": 85.0
        },
        "students": {
            "STU101": {
                "track_id": 1,
                "student_id": "STU101",
                "engagement_indicator": 85.0,
                "engagement_level": "HIGH",
                "data_quality": "HIGH",
                "contributing_factors": [
                    {"factor": "Forward-facing duration (90%)", "direction": "+", "impact": "high", "contribution": 36.0}
                ],
                "features": {
                    "visible_duration": 50.0,
                    "forward_ratio": 0.90,
                    "head_down_ratio": 0.05,
                    "phone_ratio": 0.0,
                    "sleep_ratio": 0.0,
                    "raised_hand_count": 2,
                    "raised_hand_rate": 2.4,
                    "total_event_count": 3
                }
            },
            "track_2": {
                "track_id": 2,
                "student_id": None,
                "engagement_indicator": 45.0,
                "engagement_level": "MODERATE",
                "data_quality": "LOW",
                "contributing_factors": [],
                "features": {
                    "visible_duration": 15.0,
                    "forward_ratio": 0.40,
                    "head_down_ratio": 0.30,
                    "phone_ratio": 0.20,
                    "sleep_ratio": 0.0,
                    "raised_hand_count": 0,
                    "raised_hand_rate": 0.0,
                    "total_event_count": 2
                }
            }
        },
        "timeline": [
            {
                "start": 0.0,
                "end": 60.0,
                "engagement_indicator": 85.0,
                "student_count": 1,
                "events": 3
            }
        ]
    }

    payload = {
        "lecture_id": lid,
        "report_data": engagement_report,
        "video_source": "lecture_engagement.mp4"
    }

    res = client.post("/api/processing/sessions/ingest", json=payload, headers=teacher_auth_headers)
    assert res.status_code == 201
    assert res.json()["metrics_count"] == 2
    assert res.json()["attendance_count"] == 2  # STU101 PRESENT (50s > 30s), STU102 ABSENT (enrolled but not seen)

    # Verify Engagement metrics endpoint
    eng_res = client.get(f"/api/lectures/{lid}/engagement", headers=teacher_auth_headers)
    assert eng_res.status_code == 200
    metrics = eng_res.json()
    assert len(metrics) == 2
    scored_student = [m for m in metrics if m["engagement_band"] == "HIGH"][0]
    assert scored_student["engagement_score"] == 85.0

    # Verify Engagement timeline endpoint
    tl_res = client.get(f"/api/lectures/{lid}/engagement/timeline", headers=teacher_auth_headers)
    assert tl_res.status_code == 200
    assert len(tl_res.json()) == 1
    assert tl_res.json()[0]["engagement_indicator"] == 85.0

    # Verify Attendance records: STU101 is PRESENT, STU102 is ABSENT
    att_res = client.get(f"/api/lectures/{lid}/attendance", headers=teacher_auth_headers)
    assert att_res.status_code == 200
    attendance = att_res.json()
    assert len(attendance) == 2
    status_map = {a["student"]["student_id"]: a["status"] for a in attendance}
    assert status_map["STU101"] == "PRESENT"
    assert status_map["STU102"] == "ABSENT"

    # Verify Lecture summary endpoint has aggregated data
    sum_res = client.get(f"/api/lectures/{lid}/summary", headers=teacher_auth_headers)
    assert sum_res.status_code == 200
    sum_data = sum_res.json()
    assert sum_data["total_enrolled"] == 2
    assert sum_data["total_present"] == 1
    assert sum_data["attendance_rate"] == 0.5
    assert sum_data["average_engagement"] == 65.0


def test_duplicate_ingestion_idempotency(client, teacher_auth_headers):
    """Test that ingesting the exact same report twice returns is_duplicate=True without creating duplicate records."""
    c_res = client.post("/api/classrooms", json={"name": "Idempotency Classroom"}, headers=teacher_auth_headers)
    cid = c_res.json()["id"]

    client.post("/api/students", json={"student_id": "STU_IDEMP_1", "name": "Student Idemp 1"}, headers=teacher_auth_headers)
    client.post(f"/api/classrooms/{cid}/students", json={"student_id": "STU_IDEMP_1"}, headers=teacher_auth_headers)

    l_res = client.post("/api/lectures", json={"classroom_id": cid, "title": "Idempotency Lecture"}, headers=teacher_auth_headers)
    lid = l_res.json()["id"]

    report_data = {
        "session": {"video": "idem_test.mp4", "duration": 60.0, "total_frames": 1800, "fps": 30.0},
        "students": {
            "STU_IDEMP_1": {
                "track_id": 1,
                "student_id": "STU_IDEMP_1",
                "engagement_indicator": 90.0,
                "engagement_level": "HIGH",
                "data_quality": "HIGH",
                "features": {"visible_duration": 45.0, "forward_ratio": 0.9, "raised_hand_count": 1}
            }
        },
        "timeline": [{"start": 0.0, "end": 60.0, "engagement_indicator": 90.0, "student_count": 1}]
    }

    payload = {
        "lecture_id": lid,
        "report_data": report_data,
        "video_source": "idem_test.mp4",
        "idempotency_key": "unique_idemp_key_123"
    }

    # Ingestion 1
    res1 = client.post("/api/processing/sessions/ingest", json=payload, headers=teacher_auth_headers)
    assert res1.status_code == 201
    assert res1.json()["is_duplicate"] is False
    assert res1.json()["tracks_count"] == 1
    assert res1.json()["metrics_count"] == 1
    assert res1.json()["attendance_count"] == 1

    # Ingestion 2 (Duplicate)
    res2 = client.post("/api/processing/sessions/ingest", json=payload, headers=teacher_auth_headers)
    assert res2.status_code == 201
    assert res2.json()["is_duplicate"] is True
    assert res2.json()["session_id"] == res1.json()["session_id"]

    # Verify counts have not doubled
    att_res = client.get(f"/api/lectures/{lid}/attendance", headers=teacher_auth_headers)
    assert len(att_res.json()) == 1

    eng_res = client.get(f"/api/lectures/{lid}/engagement", headers=teacher_auth_headers)
    assert len(eng_res.json()) == 1


def test_attendance_status_thresholds_present_uncertain_absent(client, teacher_auth_headers):
    """Test attendance status classification: PRESENT (>=30s), UNCERTAIN (0<t<30s), ABSENT (not seen)."""
    c_res = client.post("/api/classrooms", json={"name": "Thresholds Classroom"}, headers=teacher_auth_headers)
    cid = c_res.json()["id"]

    # Enroll 3 students
    client.post("/api/students", json={"student_id": "STU_P", "name": "Present Student"}, headers=teacher_auth_headers)
    client.post("/api/students", json={"student_id": "STU_U", "name": "Uncertain Student"}, headers=teacher_auth_headers)
    client.post("/api/students", json={"student_id": "STU_A", "name": "Absent Student"}, headers=teacher_auth_headers)

    client.post(f"/api/classrooms/{cid}/students", json={"student_id": "STU_P"}, headers=teacher_auth_headers)
    client.post(f"/api/classrooms/{cid}/students", json={"student_id": "STU_U"}, headers=teacher_auth_headers)
    client.post(f"/api/classrooms/{cid}/students", json={"student_id": "STU_A"}, headers=teacher_auth_headers)

    l_res = client.post("/api/lectures", json={"classroom_id": cid, "title": "Thresholds Lecture"}, headers=teacher_auth_headers)
    lid = l_res.json()["id"]

    report_data = {
        "session": {"video": "thresh.mp4", "duration": 60.0, "total_frames": 1800, "fps": 30.0},
        "students": {
            "STU_P": {
                "track_id": 1,
                "student_id": "STU_P",
                "features": {"visible_duration": 35.0}  # >= 30.0 -> PRESENT
            },
            "STU_U": {
                "track_id": 2,
                "student_id": "STU_U",
                "features": {"visible_duration": 15.0}  # 0 < 15 < 30 -> UNCERTAIN
            }
            # STU_A not in report -> ABSENT
        }
    }

    res = client.post("/api/processing/sessions/ingest", json={"lecture_id": lid, "report_data": report_data}, headers=teacher_auth_headers)
    assert res.status_code == 201

    att_res = client.get(f"/api/lectures/{lid}/attendance", headers=teacher_auth_headers)
    assert att_res.status_code == 200
    records = {a["student"]["student_id"]: a["status"] for a in att_res.json()}
    assert records["STU_P"] == "PRESENT"
    assert records["STU_U"] == "UNCERTAIN"
    assert records["STU_A"] == "ABSENT"


def test_behavior_only_report_populates_attendance(client, teacher_auth_headers):
    """Test that a Phase 3 report (with only per_student_events) derives attendance for enrolled students."""
    c_res = client.post("/api/classrooms", json={"name": "Behavior Attendance Classroom"}, headers=teacher_auth_headers)
    cid = c_res.json()["id"]

    client.post("/api/students", json={"student_id": "STU_BEV_1", "name": "Behavior Student 1"}, headers=teacher_auth_headers)
    client.post(f"/api/classrooms/{cid}/students", json={"student_id": "STU_BEV_1"}, headers=teacher_auth_headers)

    l_res = client.post("/api/lectures", json={"classroom_id": cid, "title": "Behavior Attendance Lecture"}, headers=teacher_auth_headers)
    lid = l_res.json()["id"]

    behavior_report = {
        "video": "behavior_att.mp4",
        "total_frames": 900,
        "average_fps": 30.0,
        "per_student_events": {
            "STU_BEV_1": [
                {
                    "track_id": 1,
                    "student_id": "STU_BEV_1",
                    "event_type": "raised_hand",
                    "duration": 35.0,  # >= 30s -> PRESENT
                    "confidence": 0.95
                }
            ]
        }
    }

    res = client.post("/api/processing/sessions/ingest", json={"lecture_id": lid, "report_data": behavior_report}, headers=teacher_auth_headers)
    assert res.status_code == 201

    att_res = client.get(f"/api/lectures/{lid}/attendance", headers=teacher_auth_headers)
    assert att_res.status_code == 200
    records = att_res.json()
    assert len(records) == 1
    assert records[0]["student"]["student_id"] == "STU_BEV_1"
    assert records[0]["status"] == "PRESENT"


def test_unenrolled_student_does_not_distort_attendance_rate(client, teacher_auth_headers):
    """Test that when an unenrolled/guest student is detected, attendance rate remains bounded by enrolled students."""
    c_res = client.post("/api/classrooms", json={"name": "Isolation Classroom"}, headers=teacher_auth_headers)
    cid = c_res.json()["id"]

    # Only 1 student enrolled: STU_ENROLLED
    client.post("/api/students", json={"student_id": "STU_ENROLLED", "name": "Enrolled Only"}, headers=teacher_auth_headers)
    client.post(f"/api/classrooms/{cid}/students", json={"student_id": "STU_ENROLLED"}, headers=teacher_auth_headers)

    l_res = client.post("/api/lectures", json={"classroom_id": cid, "title": "Isolation Lecture"}, headers=teacher_auth_headers)
    lid = l_res.json()["id"]

    # Report has STU_ENROLLED (present) AND STU_GUEST (also present, but NOT enrolled in this classroom)
    report_data = {
        "session": {"video": "guest.mp4", "duration": 60.0, "total_frames": 1800, "fps": 30.0},
        "students": {
            "STU_ENROLLED": {
                "track_id": 1,
                "student_id": "STU_ENROLLED",
                "engagement_indicator": 80.0,
                "features": {"visible_duration": 40.0}
            },
            "STU_GUEST": {
                "track_id": 2,
                "student_id": "STU_GUEST",
                "engagement_indicator": 75.0,
                "features": {"visible_duration": 45.0}
            }
        }
    }

    res = client.post("/api/processing/sessions/ingest", json={"lecture_id": lid, "report_data": report_data}, headers=teacher_auth_headers)
    assert res.status_code == 201

    sum_res = client.get(f"/api/lectures/{lid}/summary", headers=teacher_auth_headers)
    assert sum_res.status_code == 200
    sum_data = sum_res.json()
    assert sum_data["total_enrolled"] == 1
    assert sum_data["total_present"] == 1  # Only enrolled student counts as present towards enrolled
    assert sum_data["attendance_rate"] == 1.0  # 1/1 = 100%, not 2/1 = 200%!


def test_attendance_uniqueness_db_level(db_session, teacher_user):
    """Test that the attendance table strictly enforces UNIQUE(lecture_id, student_id) at DB level."""
    import pytest
    from sqlalchemy.exc import IntegrityError
    from src.backend.db.models import Classroom, Student, Lecture, Attendance

    classroom = Classroom(name="DB Uniq Room", teacher_id=teacher_user.id)
    student = Student(student_id="STU_UNIQ_DB", name="Uniq DB Student")
    db_session.add_all([classroom, student])
    db_session.commit()

    lecture = Lecture(classroom_id=classroom.id, teacher_id=teacher_user.id, title="Uniq DB Lecture")
    db_session.add(lecture)
    db_session.commit()

    att1 = Attendance(lecture_id=lecture.id, student_id=student.id, status="PRESENT")
    db_session.add(att1)
    db_session.commit()

    att2 = Attendance(lecture_id=lecture.id, student_id=student.id, status="ABSENT")
    db_session.add(att2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


