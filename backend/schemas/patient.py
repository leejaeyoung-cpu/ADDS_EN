"""
Pydantic schemas for Patient Management System
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date
from enum import Enum


class Gender(str, Enum):
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"


class PatientCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    birthdate: date
    gender: Gender
    contact: Optional[str] = None
    address: Optional[str] = None


class PatientUpdate(BaseModel):
    name: Optional[str] = None
    contact: Optional[str] = None
    address: Optional[str] = None


class PatientResponse(BaseModel):
    id: int
    patient_id: str
    name: str
    birthdate: date
    gender: str
    contact: Optional[str]
    address: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class PatientSearchResult(BaseModel):
    patients: List[PatientResponse]
    total: int
