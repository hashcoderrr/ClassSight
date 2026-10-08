from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime
from src.backend.schemas.students import StudentResponse

class AttendanceResponse(BaseModel):
    id: int
    lecture_id: int
    student_id: int
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    visible_duration: float
    status: str
    confidence: Optional[float] = None
    created_at: Optional[datetime] = None
    student: Optional[StudentResponse] = None

    model_config = ConfigDict(from_attributes=True)

class AttendanceSummary(BaseModel):
    total_students: int
    present: int
    absent: int
    uncertain: int
    attendance_rate: float
