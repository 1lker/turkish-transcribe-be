#!/usr/bin/env python3
"""Test core modules"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import logger, Config
from src.core.config import config  # config instance'ı buradan import ediyoruz
from src.core.exceptions import AudioProcessingError

def test_core():
    """Test core functionality"""
    
    # Test logger
    logger.print_banner()
    logger.info("Testing core modules...")
    
    # Test config
    logger.info(f"App name: {config.app.get('name')}")
    logger.info(f"Whisper model: {config.whisper.model_size}")
    logger.info(f"Storage path: {config.storage.base_path}")
    
    # Test creating directories
    config.storage.create_directories()
    logger.success("Directories created successfully")
    
    # Test exception
    try:
        raise AudioProcessingError(
            "Test error",
            file_path="test.mp4",
            stage="testing"
        )
    except AudioProcessingError as e:
        logger.log_error_details(e, {"test": True})
    
    logger.success("Core modules test completed!")

if __name__ == "__main__":
    test_core()