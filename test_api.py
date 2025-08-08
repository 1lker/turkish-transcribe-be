#!/usr/bin/env python3
"""Test FastAPI endpoints"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import logger

def test_api():
    """Test API functionality"""
    
    logger.print_banner()
    logger.info("Testing API modules...")
    
    try:
        from src.api import app, get_application
        from src.api.models import TranscriptionRequest, TranscriptionResponse
        
        logger.success("API modules imported successfully")
        logger.info(f"API Title: {app.title}")
        logger.info(f"API Version: {app.version}")
        logger.success("API test completed!")
        
    except Exception as e:
        logger.error(f"API test failed: {str(e)}")

if __name__ == "__main__":
    test_api()