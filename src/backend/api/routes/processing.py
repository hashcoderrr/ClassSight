from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from typing import Optional

from src.backend.db.database import get_db
from src.backend.schemas.processing import (
    ProcessingSessionCreate, ProcessingSessionResponse, IngestReportRequest, IngestReportResponse
)
from src.backend.db.models import ProcessingSession, User
from src.backend.services.ingestion_service import ingest_ai_report
from src.backend.api.dependencies import get_current_user

router = APIRouter(prefix="/api/processing/sessions", tags=["Processing"])

@router.post("", response_model=ProcessingSessionResponse, status_code=status.HTTP_201_CREATED, summary="Create a new processing session record")
def create_session(session_in: ProcessingSessionCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    session = ProcessingSession(
        lecture_id=session_in.lecture_id,
        video_source=session_in.video_source,
        status="QUEUED",
        started_at=datetime.now(timezone.utc)
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

@router.get("/{id}", response_model=ProcessingSessionResponse, summary="Get processing session details")
def get_session(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    session = db.query(ProcessingSession).filter(ProcessingSession.id == id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing session not found")
    return session

@router.get("/{id}/status", summary="Get status of processing session")
def get_session_status(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    session = db.query(ProcessingSession).filter(ProcessingSession.id == id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing session not found")
    return {
        "id": session.id,
        "status": session.status,
        "processing_fps": session.processing_fps,
        "processed_frames": session.processed_frames,
        "total_frames": session.total_frames,
        "error_message": session.error_message
    }

@router.post("/ingest", response_model=IngestReportResponse, status_code=status.HTTP_201_CREATED, summary="Ingest AI behavior/engagement JSON report")
def ingest_report(ingest_in: IngestReportRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if ingest_in.lecture_id:
        from src.backend.services.lecture_service import get_lecture_by_id
        lecture = get_lecture_by_id(db, ingest_in.lecture_id, current_user)
        if not lecture:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Lecture not found or access denied")
    try:
        result = ingest_ai_report(
            db=db,
            report_data=ingest_in.report_data,
            lecture_id=ingest_in.lecture_id,
            video_source=ingest_in.video_source,
            report_file_path=ingest_in.report_file_path,
            idempotency_key=ingest_in.idempotency_key
        )
        return IngestReportResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to ingest report: {str(e)}")

@router.post("/{id}/ingest", response_model=IngestReportResponse, status_code=status.HTTP_201_CREATED, summary="Ingest report into existing session")
def ingest_into_session(id: int, ingest_in: IngestReportRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    session = db.query(ProcessingSession).filter(ProcessingSession.id == id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing session not found")
    
    lecture_id = ingest_in.lecture_id or session.lecture_id
    if lecture_id:
        from src.backend.services.lecture_service import get_lecture_by_id
        lecture = get_lecture_by_id(db, lecture_id, current_user)
        if not lecture:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Lecture not found or access denied")
    try:
        result = ingest_ai_report(
            db=db,
            report_data=ingest_in.report_data,
            lecture_id=lecture_id,
            video_source=ingest_in.video_source or session.video_source,
            report_file_path=ingest_in.report_file_path,
            idempotency_key=ingest_in.idempotency_key
        )
        return IngestReportResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to ingest report: {str(e)}")
