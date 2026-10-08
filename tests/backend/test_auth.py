def test_register_user_success(client):
    payload = {
        "email": "newteacher@classsight.edu",
        "password": "strongpassword123",
        "full_name": "New Teacher",
        "role": "TEACHER"
    }
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newteacher@classsight.edu"
    assert data["full_name"] == "New Teacher"
    assert data["role"] == "TEACHER"
    assert "id" in data
    assert "password_hash" not in data

def test_register_duplicate_email(client, teacher_user):
    payload = {
        "email": teacher_user.email,
        "password": "anotherpassword",
        "full_name": "Duplicate Person",
        "role": "TEACHER"
    }
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]

def test_login_success(client, teacher_user):
    payload = {
        "email": teacher_user.email,
        "password": "teacher123"
    }
    response = client.post("/api/auth/login", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["role"] == "TEACHER"

def test_login_wrong_password(client, teacher_user):
    payload = {
        "email": teacher_user.email,
        "password": "wrongpassword"
    }
    response = client.post("/api/auth/login", json=payload)
    assert response.status_code == 401

def test_login_nonexistent_user(client):
    payload = {
        "email": "ghost@classsight.edu",
        "password": "somepassword"
    }
    response = client.post("/api/auth/login", json=payload)
    assert response.status_code == 401

def test_get_me_authenticated(client, teacher_auth_headers, teacher_user):
    response = client.get("/api/auth/me", headers=teacher_auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == teacher_user.email
    assert data["role"] == teacher_user.role

def test_get_me_unauthenticated(client):
    response = client.get("/api/auth/me")
    assert response.status_code == 401
