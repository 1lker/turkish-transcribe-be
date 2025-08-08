"""
Pydantic models for API requests and responses
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl, field_validator


class ModelSize(str, Enum):
    """Whisper model sizes"""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


class OutputFormat(str, Enum):
    """Output formats"""
    JSON = "json"
    TXT = "txt"
    SRT = "srt"
    VTT = "vtt"
    ALL = "all"


class ProcessingStatus(str, Enum):
    """Processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptionRequest(BaseModel):
    """Transcription request model"""
    model_config = {"protected_namespaces": ()}
    
    model_size: ModelSize = Field(default=ModelSize.BASE, description="Whisper model size")
    language: Optional[str] = Field(default="tr", description="Language code")
    device: Optional[str] = Field(default="auto", description="Device (cpu/cuda/auto)")
    apply_vad: bool = Field(default=True, description="Apply voice activity detection")
    normalize_audio: bool = Field(default=True, description="Normalize audio volume")
    output_format: OutputFormat = Field(default=OutputFormat.ALL, description="Output format")
    initial_prompt: Optional[str] = Field(default=None, description="Initial prompt for context")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Temperature for sampling")
    webhook_url: Optional[HttpUrl] = Field(default=None, description="Webhook URL for completion")


# YouTube-related models
class YouTubeInfoResponse(BaseModel):
    """YouTube video information response"""
    success: bool
    video_id: str
    title: str
    duration: Optional[int] = None  # Duration in seconds
    description: str = ""
    uploader: Optional[str] = None
    upload_date: Optional[str] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    thumbnail: Optional[str] = None
    has_audio: bool = True
    formats_count: int = 0


class YouTubeDownloadRequest(BaseModel):
    """YouTube download request"""
    url: HttpUrl
    quality: str = Field(default="best", description="Audio quality (best, high, medium)")
    session_id: Optional[str] = None


class YouTubeDownloadResponse(BaseModel):
    """YouTube download response"""
    success: bool
    session_id: str
    status: str
    message: str
    error: Optional[str] = None


class DownloadProgressUpdate(BaseModel):
    """Download progress update"""
    session_id: str
    status: str
    percent: Optional[str] = None
    speed: Optional[str] = None
    eta: Optional[str] = None
    filename: Optional[str] = None
    message: Optional[str] = None


class YouTubeFormatInfo(BaseModel):
    """YouTube format information"""
    format_id: str
    ext: str
    acodec: Optional[str] = None
    abr: Optional[float] = None  # Audio bitrate
    asr: Optional[int] = None    # Audio sample rate
    filesize: Optional[int] = None
    quality: Optional[str] = None  # Can be None, str, int, or float from yt-dlp
    format_note: str = ""
    
    @field_validator('quality', mode='before')
    @classmethod
    def validate_quality(cls, v):
        """Convert quality to string if it's not None"""
        if v is None:
            return None
        return str(v)


class YouTubeFormatsResponse(BaseModel):
    """YouTube formats response"""
    success: bool
    formats: List[YouTubeFormatInfo]
    total_formats: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_size": "base",
                "language": "tr",
                "device": "auto",
                "apply_vad": True,
                "normalize_audio": True,
                "output_format": "all"
            }
        }


class TranscriptionSegment(BaseModel):
    """Transcription segment"""
    id: int
    start: float
    end: float
    text: str
    confidence: Optional[float] = None


class TranscriptionResponse(BaseModel):
    """Transcription response model"""
    task_id: str
    status: ProcessingStatus
    text: Optional[str] = None
    segments: Optional[List[TranscriptionSegment]] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    processing_time: Optional[float] = None
    model_size: Optional[str] = None
    device: Optional[str] = None
    word_count: Optional[int] = None
    output_files: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "text": "Transcribed text here...",
                "language": "tr",
                "duration": 120.5,
                "processing_time": 45.2,
                "word_count": 250,
                "created_at": "2024-01-01T12:00:00"
            }
        }


class FileInfoResponse(BaseModel):
    """File information response"""
    filename: str
    format: str
    duration: float
    duration_minutes: float
    sample_rate: int
    channels: int
    codec: Optional[str] = None
    bit_rate: Optional[int] = None
    file_size: int
    file_size_mb: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "filename": "audio.wav",
                "format": "wav",
                "duration": 120.5,
                "duration_minutes": 2.01,
                "sample_rate": 16000,
                "channels": 1,
                "file_size": 3840000,
                "file_size_mb": 3.66
            }
        }


class LanguageDetectionResponse(BaseModel):
    """Language detection response"""
    detected_language: str
    confidence: float
    language_name: Optional[str] = None
    alternatives: Optional[List[Dict[str, float]]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "detected_language": "tr",
                "confidence": 0.95,
                "language_name": "Turkish",
                "alternatives": [
                    {"en": 0.03},
                    {"ar": 0.02}
                ]
            }
        }


class BatchTranscriptionRequest(BaseModel):
    """Batch transcription request"""
    file_ids: List[str]
    settings: TranscriptionRequest
    parallel: bool = Field(default=True, description="Process files in parallel")
    batch_size: int = Field(default=4, description="Batch size for parallel processing")


class BatchTranscriptionResponse(BaseModel):
    """Batch transcription response"""
    batch_id: str
    total_files: int
    completed: int
    failed: int
    pending: int
    results: List[TranscriptionResponse]
    created_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "batch-123",
                "total_files": 10,
                "completed": 8,
                "failed": 1,
                "pending": 1,
                "created_at": "2024-01-01T12:00:00"
            }
        }


class TaskStatusResponse(BaseModel):
    """Task status response"""
    task_id: str
    status: ProcessingStatus
    progress: Optional[float] = None
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing",
                "progress": 0.45,
                "message": "Processing audio...",
                "created_at": "2024-01-01T12:00:00",
                "updated_at": "2024-01-01T12:01:00"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: datetime
    whisper_model_loaded: bool
    available_models: List[str]
    gpu_available: bool
    gpu_name: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-01T12:00:00",
                "whisper_model_loaded": True,
                "available_models": ["base"],
                "gpu_available": False
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "File not found",
                "detail": "The requested file does not exist",
                "status_code": 404,
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str  # progress, status, result, error
    task_id: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)