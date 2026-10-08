def test_create_and_get_lecture(client, teacher_auth_headers):
    # Setup classroom
    c_res = client.post("/api/classrooms", json={"name": "Lecture Class"}, headers=teacher_auth_headers)
    cid = c_res.json()["id"]

    # Create lecture
    l_payload = {
        "classroom_id": cid,
        "title": "Deep Learning Fundamentals",
        "subject": "CS301",
        "status": "SCHEDULED"
    }
    l_res = client.post("/api/lectures", json=l_payload, headers=teacher_auth_headers)
    assert l_res.status_code == 201
    lid = l_res.json()["id"]
    assert l_res.json()["title"] == "Deep Learning Fundamentals"

    # Get lecture
    get_res = client.get(f"/api/lectures/{lid}", headers=teacher_auth_headers)
    assert get_res.status_code == 200
    assert get_res.json()["subject"] == "CS301"

def test_lecture_teacher_isolation(client, teacher_auth_headers, other_teacher_auth_headers):
    c_res = client.post("/api/classrooms", json={"name": "Teacher A Room"}, headers=teacher_auth_headers)
    cid = c_res.json()["id"]

    l_res = client.post("/api/lectures", json={"classroom_id": cid, "title": "Teacher A Lecture"}, headers=teacher_auth_headers)
    lid = l_res.json()["id"]

    # Other teacher cannot access Teacher A's lecture
    assert client.get(f"/api/lectures/{lid}", headers=other_teacher_auth_headers).status_code == 404
    # Other teacher cannot create lecture in Teacher A's classroom
    assert client.post("/api/lectures", json={"classroom_id": cid, "title": "Sneak Lecture"}, headers=other_teacher_auth_headers).status_code == 403

def test_lecture_summary_empty(client, teacher_auth_headers):
    c_res = client.post("/api/classrooms", json={"name": "Summary Class"}, headers=teacher_auth_headers)
    cid = c_res.json()["id"]

    l_res = client.post("/api/lectures", json={"classroom_id": cid, "title": "Summary Lecture"}, headers=teacher_auth_headers)
    lid = l_res.json()["id"]

    sum_res = client.get(f"/api/lectures/{lid}/summary", headers=teacher_auth_headers)
    assert sum_res.status_code == 200
    data = sum_res.json()
    assert data["total_enrolled"] == 0
    assert data["total_present"] == 0
    assert data["attendance_rate"] == 0.0
