def test_export_reports_json_and_csv(client, teacher_auth_headers):
    # Setup classroom and lecture
    c_res = client.post("/api/classrooms", json={"name": "Report Classroom"}, headers=teacher_auth_headers)
    cid = c_res.json()["id"]

    l_res = client.post("/api/lectures", json={"classroom_id": cid, "title": "Report Lecture"}, headers=teacher_auth_headers)
    lid = l_res.json()["id"]

    # Ingest a sample report
    sample_report = {
        "video": "export_test.mp4",
        "students": {
            "STU_REP_1": {
                "track_id": 1,
                "student_id": "STU_REP_1",
                "engagement_indicator": 90.0,
                "engagement_level": "HIGH",
                "data_quality": "HIGH",
                "contributing_factors": [{"factor": "Forward", "direction": "+", "impact": "high", "contribution": 40.0}],
                "features": {
                    "visible_duration": 45.0,
                    "forward_ratio": 0.95,
                    "head_down_ratio": 0.0,
                    "phone_ratio": 0.0,
                    "sleep_ratio": 0.0,
                    "raised_hand_count": 3,
                    "raised_hand_rate": 4.0,
                    "total_event_count": 3
                }
            }
        },
        "per_student_events": {
            "STU_REP_1": [
                {
                    "track_id": 1,
                    "student_id": "STU_REP_1",
                    "event_type": "raised_hand",
                    "confidence": 0.99,
                    "start_time": 5.0,
                    "end_time": 8.0,
                    "duration": 3.0
                }
            ]
        }
    }

    client.post("/api/processing/sessions/ingest", json={"lecture_id": lid, "report_data": sample_report}, headers=teacher_auth_headers)

    # 1. Behavior Report
    res_b_json = client.get(f"/api/lectures/{lid}/reports/behavior?format=json", headers=teacher_auth_headers)
    assert res_b_json.status_code == 200
    assert "events" in res_b_json.json()
    assert len(res_b_json.json()["events"]) >= 1

    res_b_csv = client.get(f"/api/lectures/{lid}/reports/behavior?format=csv", headers=teacher_auth_headers)
    assert res_b_csv.status_code == 200
    assert "text/csv" in res_b_csv.headers["content-type"]
    assert "raised_hand" in res_b_csv.text

    # 2. Engagement Report
    res_e_json = client.get(f"/api/lectures/{lid}/reports/engagement?format=json", headers=teacher_auth_headers)
    assert res_e_json.status_code == 200
    assert "metrics" in res_e_json.json()
    assert len(res_e_json.json()["metrics"]) >= 1

    res_e_csv = client.get(f"/api/lectures/{lid}/reports/engagement?format=csv", headers=teacher_auth_headers)
    assert res_e_csv.status_code == 200
    assert "text/csv" in res_e_csv.headers["content-type"]
    assert "STU_REP_1" in res_e_csv.text

    # 3. Attendance Report
    res_a_json = client.get(f"/api/lectures/{lid}/reports/attendance?format=json", headers=teacher_auth_headers)
    assert res_a_json.status_code == 200
    assert "attendance" in res_a_json.json()

    res_a_csv = client.get(f"/api/lectures/{lid}/reports/attendance?format=csv", headers=teacher_auth_headers)
    assert res_a_csv.status_code == 200
    assert "text/csv" in res_a_csv.headers["content-type"]

    # 4. Summary Report
    res_s_json = client.get(f"/api/lectures/{lid}/reports/summary?format=json", headers=teacher_auth_headers)
    assert res_s_json.status_code == 200
    assert "total_enrolled" in res_s_json.json()

    res_s_csv = client.get(f"/api/lectures/{lid}/reports/summary?format=csv", headers=teacher_auth_headers)
    assert res_s_csv.status_code == 200
    assert "text/csv" in res_s_csv.headers["content-type"]


def test_csv_injection_sanitization(client, teacher_auth_headers):
    """Test that CSV cells starting with dangerous characters (=, +, -, @) are sanitized with a leading quote."""
    c_res = client.post("/api/classrooms", json={"name": "Sanitization Classroom"}, headers=teacher_auth_headers)
    cid = c_res.json()["id"]

    # Student with potentially dangerous formula in name
    client.post("/api/students", json={"student_id": "STU_FORMULA", "name": "=cmd|' /C calc'!A0"}, headers=teacher_auth_headers)
    client.post(f"/api/classrooms/{cid}/students", json={"student_id": "STU_FORMULA"}, headers=teacher_auth_headers)

    l_res = client.post("/api/lectures", json={"classroom_id": cid, "title": "=SUM(1+1)"}, headers=teacher_auth_headers)
    lid = l_res.json()["id"]

    report_data = {
        "session": {"video": "formula.mp4", "duration": 60.0, "fps": 30.0},
        "students": {
            "STU_FORMULA": {
                "track_id": 1,
                "student_id": "STU_FORMULA",
                "features": {"visible_duration": 40.0}
            }
        }
    }
    client.post("/api/processing/sessions/ingest", json={"lecture_id": lid, "report_data": report_data}, headers=teacher_auth_headers)

    # Export attendance report
    att_csv = client.get(f"/api/lectures/{lid}/reports/attendance?format=csv", headers=teacher_auth_headers)
    assert att_csv.status_code == 200
    # Dangerous string must be prepended with single quote
    assert "'=cmd|' /C calc'!A0" in att_csv.text

    # Export summary report
    sum_csv = client.get(f"/api/lectures/{lid}/reports/summary?format=csv", headers=teacher_auth_headers)
    assert sum_csv.status_code == 200
    assert "'=SUM(1+1)" in sum_csv.text


def test_summary_report_semantic_values_in_csv(client, teacher_auth_headers):
    """Test that summary report CSV accurately formats calculated metrics (rates, engagement, N/A)."""
    c_res = client.post("/api/classrooms", json={"name": "CSV Semantics Classroom"}, headers=teacher_auth_headers)
    cid = c_res.json()["id"]

    l_res = client.post("/api/lectures", json={"classroom_id": cid, "title": "CSV Semantics Lecture"}, headers=teacher_auth_headers)
    lid = l_res.json()["id"]

    # Without ingestion, average engagement is None -> should render as N/A in CSV
    sum_csv = client.get(f"/api/lectures/{lid}/reports/summary?format=csv", headers=teacher_auth_headers)
    assert sum_csv.status_code == 200
    assert "Average Engagement,N/A" in sum_csv.text
    assert "Attendance Rate,0.0%" in sum_csv.text

