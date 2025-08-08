#!/usr/bin/env python3
"""Test transcription modules"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import logger
from src.transcription import WhisperEngine, AudioProcessor, TranscriptionPipeline

def test_transcription():
    """Test transcription functionality"""
    
    logger.print_banner()
    logger.info("Testing transcription modules...")
    
    # Test 1: WhisperEngine
    try:
        logger.info("Testing WhisperEngine...")
        engine = WhisperEngine(model_size="base", device="cpu")
        info = engine.get_model_info()
        logger.success(f"WhisperEngine loaded: {info['parameters']} parameters")
        engine.unload_model()
    except Exception as e:
        logger.error(f"WhisperEngine test failed: {str(e)}")
    
    # Test 2: AudioProcessor
    try:
        logger.info("Testing AudioProcessor...")
        processor = AudioProcessor()
        logger.success("AudioProcessor initialized")
        processor.cleanup_temp_files()
    except Exception as e:
        logger.error(f"AudioProcessor test failed: {str(e)}")
    
    # Test 3: TranscriptionPipeline
    try:
        logger.info("Testing TranscriptionPipeline...")
        pipeline = TranscriptionPipeline(model_size="base", device="cpu")
        stats = pipeline.get_statistics()
        logger.success(f"TranscriptionPipeline initialized")
        logger.info(f"Stats: {stats}")
        pipeline.cleanup()
    except Exception as e:
        logger.error(f"TranscriptionPipeline test failed: {str(e)}")
    
    logger.success("Transcription module tests completed!")

if __name__ == "__main__":
    test_transcription()