#!/usr/bin/env python3
"""Phase 5.1 End-to-End Verification Script:

Deterministic Semantic Verification & Ingestion Idempotency
Classroom / Enrolled Students -> Deterministic AI Report -> Backend Ingestion -> Database -> REST APIs
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from fastapi.testclient import TestClient
from src.backend.main import app


def run_end_to_end_verification():
    print("=" * 70)
    print("ClassSight AI - Phase 5.1 End-to-End Backend Verification")
    print("=" * 70)

    client = TestClient(app)

    # 1. Health Check
    print("\n[Step 1] Verifying System & Database Health...")
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health_data = res.json()
    assert health_data["status"] == "ok"
    assert health_data["database"] == "connected"
    print(f"  Health OK: database={health_data['database']}, version={health_data.get('version')}")

    # 2. Authentication: Register and Login
    print("\n[Step 2] Authenticating Teacher Account...")
    teacher_email = "prof.ada.lovelace@classsight.edu"
    reg_payload = {
        "email": teacher_email,
        "password": "SecurePassword123!",
        "full_name": "Prof. Ada Lovelace",
        "role": "TEACHER",
    }
    client.post("/api/auth/register", json=reg_payload)
    login_res = client.post(
        "/api/auth/login",
        json={"email": teacher_email, "password": "SecurePassword123!"},
    )
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print(f"  Authenticated successfully as {login_res.json()['full_name']} ({login_res.json()['role']})")

    # 3. Create Classroom
    print("\n[Step 3] Creating Classroom...")
    c_res = client.post(
        "/api/classrooms",
        json={
            "name": "Machine Learning & Perception (CS901)",
            "room_number": "Auditorium 1",
            "description": "Graduate level perception and behavior analytics.",
        },
        headers=headers,
    )
    assert c_res.status_code in [200, 201], f"Failed to create classroom: {c_res.text}"
    classroom_id = c_res.json()["id"]
    print(f"  Created Classroom ID: {classroom_id} ('{c_res.json()['name']}')")

    # 4. Enroll Students
    print("\n[Step 4] Enrolling Students...")
    enrolled_students = [
        ("STU001", "Alex Rivera"),
        ("STU002", "Maya Lin"),
        ("STU003", "Samir Patel"),
    ]
    for sid, sname in enrolled_students:
        client.post(
            "/api/students",
            json={"student_id": sid, "name": sname},
            headers=headers,
        )
        client.post(
            f"/api/classrooms/{classroom_id}/students",
            json={"student_id": sid},
            headers=headers,
        )

    students_res = client.get(
        f"/api/classrooms/{classroom_id}/students", headers=headers
    )
    assert students_res.status_code == 200
    assert len(students_res.json()) == 3, f"Expected 3 enrolled students, got {len(students_res.json())}"
    print(f"  Classroom enrollment verified: exactly 3 students enrolled.")

    # 5. Create Lecture
    print("\n[Step 5] Scheduling Lecture...")
    l_res = client.post(
        "/api/lectures",
        json={
            "classroom_id": classroom_id,
            "title": "Lecture 1: Real-time Attention Modeling",
            "subject": "CS901",
            "status": "COMPLETED",
        },
        headers=headers,
    )
    assert l_res.status_code == 201, f"Failed to create lecture: {l_res.text}"
    lecture_id = l_res.json()["id"]
    print(f"  Created Lecture ID: {lecture_id} ('{l_res.json()['title']}')")

    # 6. Ingest Deterministic AI Report
    print("\n[Step 6] Ingesting Deterministic AI Report into Database...")
    deterministic_report = {
        "session": {
            "video": "lecture1_session.mp4",
            "fps": 30.0,
            "total_frames": 2700,
            "duration": 90.0,
        },
        "students": {
            "STU001": {
                "track_id": 1,
                "student_id": "STU001",
                "engagement_indicator": 82.0,
                "engagement_level": "HIGH",
                "data_quality": "HIGH",
                "contributing_factors": [
                    {
                        "factor": "Forward-facing duration (88%)",
                        "direction": "+",
                        "impact": "high",
                        "contribution": 35.2,
                    }
                ],
                "features": {
                    "visible_duration": 85.0,  # >= 30.0s -> PRESENT
                    "forward_ratio": 0.88,
                    "head_down_ratio": 0.05,
                    "phone_ratio": 0.0,
                    "sleep_ratio": 0.0,
                    "raised_hand_count": 3,
                    "raised_hand_rate": 2.1,
                    "total_event_count": 4,
                },
            },
            "STU002": {
                "track_id": 2,
                "student_id": "STU002",
                "engagement_indicator": 55.0,
                "engagement_level": "MODERATE",
                "data_quality": "MEDIUM",
                "contributing_factors": [
                    {
                        "factor": "Head-down posture (20%)",
                        "direction": "-",
                        "impact": "moderate",
                        "contribution": -2.0,
                    }
                ],
                "features": {
                    "visible_duration": 65.0,  # >= 30.0s -> PRESENT
                    "forward_ratio": 0.60,
                    "head_down_ratio": 0.20,
                    "phone_ratio": 0.10,
                    "sleep_ratio": 0.0,
                    "raised_hand_count": 0,
                    "raised_hand_rate": 0.0,
                    "total_event_count": 2,
                },
            },
            "track_5": {
                "track_id": 5,
                "student_id": None,
                "engagement_indicator": 40.0,
                "engagement_level": "MODERATE",
                "data_quality": "LOW",
                "contributing_factors": [],
                "features": {
                    "visible_duration": 12.0,
                    "forward_ratio": 0.50,
                    "head_down_ratio": 0.30,
                    "phone_ratio": 0.20,
                    "sleep_ratio": 0.0,
                    "raised_hand_count": 0,
                    "raised_hand_rate": 0.0,
                    "total_event_count": 1,
                },
            },
        },
        "timeline": [
            {
                "start": 0.0,
                "end": 45.0,
                "engagement_indicator": 75.0,
                "student_count": 2,
                "events": 3,
            },
            {
                "start": 45.0,
                "end": 90.0,
                "engagement_indicator": 62.0,
                "student_count": 2,
                "events": 4,
            },
        ],
    }

    ingest_payload = {
        "lecture_id": lecture_id,
        "report_data": deterministic_report,
        "video_source": "lecture1_session.mp4",
        "idempotency_key": f"lecture_{lecture_id}_run_1",
    }

    ingest_res = client.post(
        "/api/processing/sessions/ingest", json=ingest_payload, headers=headers
    )
    assert ingest_res.status_code == 201, f"Ingestion failed: {ingest_res.text}"
    summary1 = ingest_res.json()
    assert summary1["is_duplicate"] is False, "Initial ingestion should not be duplicate"
    assert summary1["tracks_count"] == 3, f"Expected 3 tracks, got {summary1['tracks_count']}"
    assert summary1["metrics_count"] == 3, f"Expected 3 metrics, got {summary1['metrics_count']}"
    assert summary1["attendance_count"] == 3, f"Expected 3 attendance records, got {summary1['attendance_count']}"
    print(f"  Ingestion 1 Successful: session_id={summary1['session_id']}, tracks={summary1['tracks_count']}, metrics={summary1['metrics_count']}, attendance={summary1['attendance_count']}")

    # 7. Semantic Assertions on Analytics APIs
    print("\n[Step 7] Verifying Semantic Analytics & Attendance Correctness...")

    # Attendance Verification
    att_res = client.get(f"/api/lectures/{lecture_id}/attendance", headers=headers)
    assert att_res.status_code == 200
    att_records = att_res.json()
    assert len(att_records) == 3, f"Expected exactly 3 attendance records, got {len(att_records)}"

    status_by_sid = {a["student"]["student_id"]: a["status"] for a in att_records}
    assert status_by_sid["STU001"] == "PRESENT", f"STU001 should be PRESENT (85s visible), got {status_by_sid.get('STU001')}"
    assert status_by_sid["STU002"] == "PRESENT", f"STU002 should be PRESENT (65s visible), got {status_by_sid.get('STU002')}"
    assert status_by_sid["STU003"] == "ABSENT", f"STU003 should be ABSENT (not seen), got {status_by_sid.get('STU003')}"
    print("  Attendance Semantics: STU001=PRESENT, STU002=PRESENT, STU003=ABSENT (VERIFIED)")

    # Engagement Verification
    eng_res = client.get(f"/api/lectures/{lecture_id}/engagement", headers=headers)
    assert eng_res.status_code == 200
    eng_metrics = eng_res.json()
    assert len(eng_metrics) == 3, f"Expected 3 engagement metrics, got {len(eng_metrics)}"

    # Timeline Verification
    tl_res = client.get(f"/api/lectures/{lecture_id}/engagement/timeline", headers=headers)
    assert tl_res.status_code == 200
    assert len(tl_res.json()) == 2, f"Expected 2 timeline windows, got {len(tl_res.json())}"

    # Lecture Analytical Summary Verification
    sum_res = client.get(f"/api/lectures/{lecture_id}/summary", headers=headers)
    assert sum_res.status_code == 200
    sum_data = sum_res.json()
    assert sum_data["total_enrolled"] == 3, f"Expected 3 enrolled, got {sum_data['total_enrolled']}"
    assert sum_data["total_present"] == 2, f"Expected 2 present, got {sum_data['total_present']}"
    expected_rate = round(2 / 3, 4)
    assert sum_data["attendance_rate"] == expected_rate, f"Expected rate {expected_rate}, got {sum_data['attendance_rate']}"
    # Scored students: STU001 (82.0), STU002 (55.0), track_5 (40.0). Average = (82 + 55 + 40) / 3 = 59.0
    assert sum_data["average_engagement"] is not None, "Average engagement should not be None"
    assert sum_data["average_engagement"] == 59.0, f"Expected average engagement 59.0, got {sum_data['average_engagement']}"
    print(f"  Summary Semantics: enrolled=3, present=2, rate={sum_data['attendance_rate']*100:.1f}%, avg_engagement={sum_data['average_engagement']} (VERIFIED)")

    # 8. Ingestion Idempotency Verification
    print("\n[Step 8] Verifying Ingestion Idempotency (Duplicate Prevention)...")
    ingest_res2 = client.post(
        "/api/processing/sessions/ingest", json=ingest_payload, headers=headers
    )
    assert ingest_res2.status_code == 201, f"Repeated ingestion call failed: {ingest_res2.text}"
    summary2 = ingest_res2.json()
    assert summary2["is_duplicate"] is True, f"Expected is_duplicate=True on re-ingestion, got {summary2}"
    assert summary2["session_id"] == summary1["session_id"], "Duplicate ingestion must reference the same session ID"
    print(f"  Idempotency Flag: is_duplicate={summary2['is_duplicate']}, message='{summary2['message']}'")

    # Assert database counts have not doubled
    att_res_after = client.get(f"/api/lectures/{lecture_id}/attendance", headers=headers)
    assert len(att_res_after.json()) == 3, f"Duplicate ingestion multiplied attendance records: {len(att_res_after.json())}"

    sum_res_after = client.get(f"/api/lectures/{lecture_id}/summary", headers=headers)
    sum_data_after = sum_res_after.json()
    assert sum_data_after["total_present"] == 2, "Duplicate ingestion corrupted present count"
    assert sum_data_after["attendance_rate"] == expected_rate, "Duplicate ingestion corrupted attendance rate"
    print("  Idempotency Verification: Records count preserved; no duplicate tracks, metrics, or attendance.")

    # 9. Report Export Verification
    print("\n[Step 9] Verifying Report Exports & CSV Formatting...")
    csv_res = client.get(f"/api/lectures/{lecture_id}/reports/summary?format=csv", headers=headers)
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers["content-type"]
    csv_text = csv_res.text
    assert "Metric,Value" in csv_text
    assert "Total Enrolled,3" in csv_text
    assert "Total Present,2" in csv_text
    assert "Attendance Rate,66.7%" in csv_text
    assert "Average Engagement,59.0" in csv_text
    print("  Report Export: CSV Summary accurately reflects computed analytical values.")

    print("\n" + "=" * 70)
    print("ALL PHASE 5.1 END-TO-END SEMANTIC VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_end_to_end_verification()
