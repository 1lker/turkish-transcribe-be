"""
Core module for Turkish Education Transcription System
"""

from .config import settings, Config
from .logger import logger, setup_logger
from .exceptions import (
    TranscriptionError,
    AudioProcessingError,
    ModelLoadError,
    ValidationError,
    StorageError
)

__all__ = [
    'settings',
    'Config',
    'logger',
    'setup_logger',
    'TranscriptionError',
    'AudioProcessingError', 
    'ModelLoadError',
    'ValidationError',
    'StorageError'
]

__version__ = "1.0.0"