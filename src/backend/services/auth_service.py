from sqlalchemy.orm import Session
from typing import Optional
from src.backend.db.models import User
from src.backend.schemas.auth import UserRegister, UserLogin
from src.backend.core.security import get_password_hash, verify_password

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email.lower().strip()).first()

def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()

def register_user(db: Session, user_in: UserRegister) -> User:
    existing = get_user_by_email(db, user_in.email)
    if existing:
        raise ValueError(f"User with email '{user_in.email}' already exists")
    
    hashed_pw = get_password_hash(user_in.password)
    role = user_in.role.upper() if user_in.role else "TEACHER"
    if role not in ["ADMIN", "TEACHER"]:
        role = "TEACHER"

    user = User(
        email=user_in.email.lower().strip(),
        password_hash=hashed_pw,
        full_name=user_in.full_name,
        role=role,
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def authenticate_user(db: Session, login_in: UserLogin) -> Optional[User]:
    user = get_user_by_email(db, login_in.email)
    if not user:
        return None
    if not user.is_active:
        return None
    if not verify_password(login_in.password, user.password_hash):
        return None
    return user
