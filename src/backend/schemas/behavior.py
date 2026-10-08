from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict
from datetime import datetime

class BehaviorEventResponse(BaseModel):
    id: int
    processing_session_id: int
    track_id: int
    student_id: Optional[int] = None
    event_type: str
    confidence: Optional[float] = None
    start_time: float
    end_time: float
    duration: float
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class BehaviorSummary(BaseModel):
    total_events: int
    event_counts: Dict[str, int]
    durations_by_type: Dict[str, float]
