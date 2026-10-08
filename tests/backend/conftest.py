import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.backend.db.database import Base, get_db
from src.backend.main import app
from src.backend.core.security import create_access_token, get_password_hash
from src.backend.db.models import User, Classroom, Student, ClassroomStudent, Lecture

# In-memory SQLite for fast, isolated testing
TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def teacher_user(db_session):
    user = User(
        email="teacher@classsight.edu",
        password_hash=get_password_hash("teacher123"),
        full_name="Prof. Alan Turing",
        role="TEACHER",
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture(scope="function")
def admin_user(db_session):
    user = User(
        email="admin@classsight.edu",
        password_hash=get_password_hash("admin123"),
        full_name="Dean Ada Lovelace",
        role="ADMIN",
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture(scope="function")
def other_teacher(db_session):
    user = User(
        email="other@classsight.edu",
        password_hash=get_password_hash("other123"),
        full_name="Prof. Grace Hopper",
        role="TEACHER",
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture(scope="function")
def teacher_auth_headers(teacher_user):
    token = create_access_token({"sub": teacher_user.email, "role": teacher_user.role})
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture(scope="function")
def admin_auth_headers(admin_user):
    token = create_access_token({"sub": admin_user.email, "role": admin_user.role})
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture(scope="function")
def other_teacher_auth_headers(other_teacher):
    token = create_access_token({"sub": other_teacher.email, "role": other_teacher.role})
    return {"Authorization": f"Bearer {token}"}
