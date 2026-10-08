from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from src.backend.db.database import get_db
from src.backend.schemas.users import UserResponse
from src.backend.db.models import User
from src.backend.api.dependencies import require_admin

router = APIRouter(prefix="/api/users", tags=["Users"])

@router.get("", response_model=List[UserResponse], summary="List all users (Admin only)")
def list_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(require_admin)):
    return db.query(User).offset(skip).limit(limit).all()

@router.get("/{id}", response_model=UserResponse, summary="Get user details by ID (Admin only)")
def get_user(id: int, db: Session = Depends(get_db), current_user: User = Depends(require_admin)):
    user = db.query(User).filter(User.id == id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user
