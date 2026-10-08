from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Optional
from datetime import datetime

class StudentBase(BaseModel):
    student_id: str
    name: str
    email: Optional[EmailStr] = None
    roll_number: Optional[str] = None
    is_active: bool = True

class StudentCreate(StudentBase):
    pass

class StudentUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    roll_number: Optional[str] = None
    is_active: Optional[bool] = None

class StudentResponse(StudentBase):
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
