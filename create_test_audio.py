#!/usr/bin/env python3
"""Create a test audio file"""

import numpy as np
import soundfile as sf
from pathlib import Path

def create_test_audio():
    """Create a simple test audio file with sine wave"""
    
    # Parameters
    duration = 5  # seconds
    sample_rate = 16000
    frequency = 440  # Hz (A4 note)
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate sine wave
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Add some silence
    silence = np.zeros(int(sample_rate * 0.5))
    audio = np.concatenate([silence, audio, silence])
    
    # Save as WAV
    output_path = Path("test_audio.wav")
    sf.write(output_path, audio, sample_rate)
    
    print(f"Test audio file created: {output_path}")
    print(f"Duration: {len(audio) / sample_rate:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")

if __name__ == "__main__":
    create_test_audio()