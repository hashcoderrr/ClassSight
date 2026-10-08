from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime

class LectureBase(BaseModel):
    classroom_id: int
    title: str
    subject: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: Optional[str] = "SCHEDULED"

class LectureCreate(LectureBase):
    pass

class LectureUpdate(BaseModel):
    title: Optional[str] = None
    subject: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: Optional[str] = None

class LectureResponse(LectureBase):
    id: int
    teacher_id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class LectureSummary(BaseModel):
    lecture: LectureResponse
    total_enrolled: int
    total_present: int
    attendance_rate: float
    average_engagement: Optional[float] = None
    total_behavior_events: int = 0
