from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime

class ProcessingSessionCreate(BaseModel):
    lecture_id: Optional[int] = None
    video_source: Optional[str] = None

class ProcessingSessionResponse(BaseModel):
    id: int
    lecture_id: Optional[int] = None
    video_source: Optional[str] = None
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_fps: Optional[float] = None
    total_frames: Optional[int] = None
    processed_frames: Optional[int] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class IngestReportRequest(BaseModel):
    lecture_id: Optional[int] = None
    report_data: Dict[str, Any]
    video_source: Optional[str] = None
    report_file_path: Optional[str] = None
    idempotency_key: Optional[str] = None

class IngestReportResponse(BaseModel):
    session_id: int
    lecture_id: Optional[int] = None
    status: str
    tracks_count: int
    events_count: int
    metrics_count: int
    attendance_count: int
    message: str
    is_duplicate: bool = False

