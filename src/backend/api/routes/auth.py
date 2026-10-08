from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.backend.db.database import get_db
from src.backend.schemas.auth import UserRegister, UserLogin, Token
from src.backend.schemas.users import UserResponse
from src.backend.services.auth_service import register_user, authenticate_user
from src.backend.core.security import create_access_token
from src.backend.api.dependencies import get_current_user
from src.backend.db.models import User

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED, summary="Register a new user account")
def register(user_in: UserRegister, db: Session = Depends(get_db)):
    try:
        user = register_user(db, user_in)
        return user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/login", response_model=Token, summary="Authenticate and retrieve JWT token")
def login(login_in: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(db, login_in)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={"sub": user.email, "role": user.role})
    return Token(access_token=token, token_type="bearer", role=user.role, full_name=user.full_name)

@router.get("/me", response_model=UserResponse, summary="Get current logged-in user profile")
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
