from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.backend.db.database import get_db

router = APIRouter(tags=["Health"])

@router.get("/health", summary="Basic service health check")
def health_check(db: Session = Depends(get_db)):
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"
    return {
        "status": "ok",
        "database": db_status,
        "version": "1.0.0"
    }

@router.get("/api/health", summary="API health check")
def api_health(db: Session = Depends(get_db)):
    return health_check(db)
