"""
Pydantic models defining the request and response schemas for the API.
"""

from pydantic import BaseModel
from typing import Optional


class HealthResponse(BaseModel):
    status: str
    message: str


class DetectionResponse(BaseModel):
    success: bool
    filename: str
    file_hash: str
    classification: str           # "REAL" or "FAKE"
    fake_probability: float
    real_probability: float
    confidence: float
    frames_analysed: int
    faces_detected: int
    aggregation_method: str
    processing_time_seconds: float


class RegisterResponse(BaseModel):
    success: bool
    filename: str
    file_hash: str
    transaction_hash: Optional[str] = None
    block_number: Optional[int] = None
    block_hash: Optional[str] = None
    pending_count: Optional[int] = None
    transactions_until_block: Optional[int] = None
    message: str
    blockchain_network: str


class VerifyResponse(BaseModel):
    file_hash: str
    is_registered: bool
    registered_at: Optional[str] = None
    original_filename: Optional[str] = None
    block_number: Optional[int] = None
    block_hash: Optional[str] = None
    is_confirmed: Optional[bool] = None
    blockchain_network: Optional[str] = None
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
