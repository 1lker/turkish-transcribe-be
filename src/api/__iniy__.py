"""
API module for Turkish Education Transcription System
"""

from .app import app, get_application
from .models import (
    TranscriptionRequest,
    TranscriptionResponse,
    FileInfoResponse,
    LanguageDetectionResponse,
    HealthResponse
)

__all__ = [
    'app',
    'get_application',
    'TranscriptionRequest',
    'TranscriptionResponse',
    'FileInfoResponse',
    'LanguageDetectionResponse',
    'HealthResponse'
]

__version__ = "1.0.0"