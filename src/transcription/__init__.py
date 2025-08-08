"""
Transcription module for Turkish Education Transcription System
"""

from .whisper_engine import WhisperEngine, WhisperResult
from .audio_processor import AudioProcessor
from .transcription_pipeline import TranscriptionPipeline

__all__ = [
    'WhisperEngine',
    'WhisperResult', 
    'AudioProcessor',
    'TranscriptionPipeline'
]

__version__ = "1.0.0"