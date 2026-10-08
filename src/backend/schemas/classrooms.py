from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime
from src.backend.schemas.students import StudentResponse

class ClassroomBase(BaseModel):
    name: str
    room_number: Optional[str] = None
    description: Optional[str] = None

class ClassroomCreate(ClassroomBase):
    pass

class ClassroomUpdate(BaseModel):
    name: Optional[str] = None
    room_number: Optional[str] = None
    description: Optional[str] = None

class ClassroomResponse(ClassroomBase):
    id: int
    teacher_id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class ClassroomStudentAdd(BaseModel):
    student_id: str

class ClassroomStudentResponse(BaseModel):
    id: int
    classroom_id: int
    student_id: int
    enrolled_at: Optional[datetime] = None
    is_active: bool
    student: Optional[StudentResponse] = None

    model_config = ConfigDict(from_attributes=True)
