def test_create_student_success(client, teacher_auth_headers):
    payload = {
        "student_id": "STU101",
        "name": "Alice Smith",
        "email": "alice@classsight.edu",
        "roll_number": "CS-01"
    }
    response = client.post("/api/students", json=payload, headers=teacher_auth_headers)
    assert response.status_code == 201
    data = response.json()
    assert data["student_id"] == "STU101"
    assert data["name"] == "Alice Smith"
    assert data["is_active"] is True

def test_create_duplicate_student(client, teacher_auth_headers):
    payload = {
        "student_id": "STU102",
        "name": "Bob Jones"
    }
    res1 = client.post("/api/students", json=payload, headers=teacher_auth_headers)
    assert res1.status_code == 201
    res2 = client.post("/api/students", json=payload, headers=teacher_auth_headers)
    assert res2.status_code == 400
    assert "already exists" in res2.json()["detail"]

def test_get_students_list(client, teacher_auth_headers):
    client.post("/api/students", json={"student_id": "STU201", "name": "Charlie"}, headers=teacher_auth_headers)
    client.post("/api/students", json={"student_id": "STU202", "name": "David"}, headers=teacher_auth_headers)
    
    response = client.get("/api/students", headers=teacher_auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2

def test_get_student_by_id(client, teacher_auth_headers):
    res = client.post("/api/students", json={"student_id": "STU301", "name": "Eva"}, headers=teacher_auth_headers)
    sid = res.json()["id"]

    response = client.get(f"/api/students/{sid}", headers=teacher_auth_headers)
    assert response.status_code == 200
    assert response.json()["student_id"] == "STU301"

def test_update_student(client, teacher_auth_headers):
    res = client.post("/api/students", json={"student_id": "STU401", "name": "Frank"}, headers=teacher_auth_headers)
    sid = res.json()["id"]

    patch_res = client.patch(f"/api/students/{sid}", json={"name": "Franklin", "roll_number": "R-401"}, headers=teacher_auth_headers)
    assert patch_res.status_code == 200
    data = patch_res.json()
    assert data["name"] == "Franklin"
    assert data["roll_number"] == "R-401"

def test_delete_student(client, teacher_auth_headers):
    res = client.post("/api/students", json={"student_id": "STU501", "name": "Grace"}, headers=teacher_auth_headers)
    sid = res.json()["id"]

    del_res = client.delete(f"/api/students/{sid}", headers=teacher_auth_headers)
    assert del_res.status_code == 204

    get_res = client.get(f"/api/students/{sid}", headers=teacher_auth_headers)
    assert get_res.status_code == 404

def test_student_metrics_endpoints_empty(client, teacher_auth_headers):
    res = client.post("/api/students", json={"student_id": "STU601", "name": "Hannah"}, headers=teacher_auth_headers)
    sid = res.json()["id"]

    assert client.get(f"/api/students/{sid}/attendance", headers=teacher_auth_headers).status_code == 200
    assert client.get(f"/api/students/{sid}/behavior", headers=teacher_auth_headers).status_code == 200
    assert client.get(f"/api/students/{sid}/engagement", headers=teacher_auth_headers).status_code == 200
