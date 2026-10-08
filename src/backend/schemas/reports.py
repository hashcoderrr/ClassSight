from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime

class ReportResponse(BaseModel):
    id: int
    processing_session_id: int
    report_type: str
    file_path: Optional[str] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
