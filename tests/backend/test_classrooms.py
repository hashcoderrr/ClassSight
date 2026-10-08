def test_create_and_get_classroom(client, teacher_auth_headers):
    payload = {
        "name": "Computer Vision 101",
        "room_number": "Room 402",
        "description": "Intro to CV with OpenCV and YOLO"
    }
    response = client.post("/api/classrooms", json=payload, headers=teacher_auth_headers)
    assert response.status_code == 201
    cid = response.json()["id"]

    get_res = client.get(f"/api/classrooms/{cid}", headers=teacher_auth_headers)
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Computer Vision 101"

def test_classroom_teacher_isolation(client, teacher_auth_headers, other_teacher_auth_headers):
    # Teacher A creates classroom
    res = client.post("/api/classrooms", json={"name": "Teacher A Room"}, headers=teacher_auth_headers)
    cid = res.json()["id"]

    # Teacher B tries to access Teacher A's classroom
    get_res = client.get(f"/api/classrooms/{cid}", headers=other_teacher_auth_headers)
    assert get_res.status_code == 404

    # Teacher B list classrooms should not include Teacher A's
    list_res = client.get("/api/classrooms", headers=other_teacher_auth_headers)
    assert not any(c["id"] == cid for c in list_res.json())

def test_admin_can_access_all_classrooms(client, teacher_auth_headers, admin_auth_headers):
    res = client.post("/api/classrooms", json={"name": "Teacher A Math"}, headers=teacher_auth_headers)
    cid = res.json()["id"]

    # Admin accesses Teacher A's room
    get_res = client.get(f"/api/classrooms/{cid}", headers=admin_auth_headers)
    assert get_res.status_code == 200

def test_classroom_student_enrollment(client, teacher_auth_headers):
    # Create student
    st_res = client.post("/api/students", json={"student_id": "STU_ENROLL_1", "name": "Student Enrolled"}, headers=teacher_auth_headers)
    sid_str = st_res.json()["student_id"]
    sid_int = st_res.json()["id"]

    # Create classroom
    c_res = client.post("/api/classrooms", json={"name": "Enrollment Class"}, headers=teacher_auth_headers)
    cid = c_res.json()["id"]

    # Enroll student
    enroll_res = client.post(f"/api/classrooms/{cid}/students", json={"student_id": sid_str}, headers=teacher_auth_headers)
    assert enroll_res.status_code == 201

    # Check student list in classroom
    list_res = client.get(f"/api/classrooms/{cid}/students", headers=teacher_auth_headers)
    assert list_res.status_code == 200
    assert len(list_res.json()) == 1

    # Unenroll student
    del_res = client.delete(f"/api/classrooms/{cid}/students/{sid_int}", headers=teacher_auth_headers)
    assert del_res.status_code == 204

    # Verify empty list
    list_res2 = client.get(f"/api/classrooms/{cid}/students", headers=teacher_auth_headers)
    assert len(list_res2.json()) == 0

def test_classroom_summaries_empty(client, teacher_auth_headers):
    c_res = client.post("/api/classrooms", json={"name": "Empty Class"}, headers=teacher_auth_headers)
    cid = c_res.json()["id"]

    assert client.get(f"/api/classrooms/{cid}/attendance/summary", headers=teacher_auth_headers).status_code == 200
    assert client.get(f"/api/classrooms/{cid}/behavior/summary", headers=teacher_auth_headers).status_code == 200
    assert client.get(f"/api/classrooms/{cid}/engagement/summary", headers=teacher_auth_headers).status_code == 200
