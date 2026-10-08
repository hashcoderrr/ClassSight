from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime

class EngagementMetricResponse(BaseModel):
    id: int
    processing_session_id: int
    student_id: Optional[int] = None
    track_id: Optional[int] = None
    visible_duration: float
    forward_ratio: float
    head_down_ratio: float
    phone_ratio: float
    raised_hand_count: int
    raised_hand_rate: float
    sleep_ratio: float
    engagement_score: Optional[float] = None
    engagement_band: str
    data_quality: str
    contributing_factors: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class EngagementTimelineResponse(BaseModel):
    id: int
    processing_session_id: int
    window_start: float
    window_end: float
    engagement_indicator: Optional[float] = None
    student_count: int
    participation_count: int
    phone_event_count: int
    head_down_duration: float
    sleep_event_count: int

    model_config = ConfigDict(from_attributes=True)

class EngagementSummary(BaseModel):
    average_engagement: float
    median_engagement: float
    distribution: Dict[str, int]
    total_scored: int
    insufficient_data_count: int
