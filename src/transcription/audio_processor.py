"""
Audio processing utilities for transcription
"""

import os
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

import ffmpeg
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from src.core import logger
from src.core.config import config
from src.core.exceptions import AudioProcessingError, ValidationError


@dataclass
class AudioInfo:
    """Audio file information"""
    file_path: Path
    duration: float  # seconds
    sample_rate: int
    channels: int
    bit_rate: Optional[int]
    codec: Optional[str]
    format: str
    file_size: int  # bytes
    
    @property
    def duration_minutes(self) -> float:
        """Duration in minutes"""
        return self.duration / 60
    
    @property
    def file_size_mb(self) -> float:
        """File size in MB"""
        return self.file_size / (1024 * 1024)


class AudioProcessor:
    """Audio processing for transcription"""
    
    def __init__(self):
        """Initialize audio processor"""
        self.audio_config = config.audio
        self.temp_dir = config.storage.temp_path
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Check ffmpeg availability
        self._check_ffmpeg()
        
        logger.info("AudioProcessor initialized")
    
    def _check_ffmpeg(self):
        """Check if ffmpeg is available"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode != 0:
                raise AudioProcessingError("ffmpeg not found or not working properly")
            
            # Get ffmpeg version
            version_line = result.stdout.split('\n')[0]
            logger.debug(f"ffmpeg available: {version_line}")
            
        except FileNotFoundError:
            raise AudioProcessingError(
                "ffmpeg not found. Please install ffmpeg: "
                "brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)"
            )
    
    def extract_audio_from_video(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        audio_format: str = "wav"
    ) -> Path:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            output_path: Output audio file path
            audio_format: Output audio format
            
        Returns:
            Path to extracted audio file
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise AudioProcessingError(f"Video file not found: {video_path}")
        
        # Determine output path
        if output_path is None:
            output_path = self.temp_dir / f"{video_path.stem}.{audio_format}"
        else:
            output_path = Path(output_path)
        
        logger.info(f"Extracting audio from: {video_path.name}")
        
        try:
            # Extract audio using ffmpeg-python
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec='pcm_s16le' if audio_format == 'wav' else 'libmp3lame',
                ac=self.audio_config.channels,
                ar=self.audio_config.sample_rate,
                loglevel='error'
            )
            ffmpeg.run(stream, overwrite_output=True)
            
            logger.success(f"Audio extracted: {output_path.name}")
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise AudioProcessingError(
                f"Failed to extract audio: {error_msg}",
                file_path=str(video_path),
                stage="audio_extraction"
            )
    
    def convert_audio_format(
        self,
        input_path: Path,
        output_format: str = "wav",
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Convert audio to different format
        
        Args:
            input_path: Input audio file
            output_format: Target format
            output_path: Output file path
            
        Returns:
            Path to converted audio file
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise AudioProcessingError(f"Audio file not found: {input_path}")
        
        # Check if already in target format
        if input_path.suffix[1:] == output_format:
            logger.info(f"Audio already in {output_format} format")
            return input_path
        
        # Determine output path
        if output_path is None:
            output_path = self.temp_dir / f"{input_path.stem}.{output_format}"
        else:
            output_path = Path(output_path)
        
        logger.info(f"Converting audio: {input_path.name} -> {output_format}")
        
        try:
            # Load and convert using pydub
            audio = AudioSegment.from_file(str(input_path))
            
            # Set parameters
            audio = audio.set_frame_rate(self.audio_config.sample_rate)
            audio = audio.set_channels(self.audio_config.channels)
            
            # Export
            audio.export(
                str(output_path),
                format=output_format,
                parameters=["-q:a", "0"]  # Highest quality
            )
            
            logger.success(f"Audio converted: {output_path.name}")
            return output_path
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to convert audio: {str(e)}",
                file_path=str(input_path),
                stage="format_conversion"
            )
    
    def split_audio_into_chunks(
        self,
        audio_path: Path,
        chunk_length: Optional[int] = None,
        overlap: Optional[int] = None,
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Split audio into chunks
        
        Args:
            audio_path: Path to audio file
            chunk_length: Chunk length in seconds
            overlap: Overlap between chunks in seconds
            output_dir: Directory to save chunks
            
        Returns:
            List of chunk file paths
        """
        audio_path = Path(audio_path)
        chunk_length = chunk_length or self.audio_config.chunk_length
        overlap = overlap or self.audio_config.overlap
        
        if not audio_path.exists():
            raise AudioProcessingError(f"Audio file not found: {audio_path}")
        
        # Create output directory
        if output_dir is None:
            output_dir = self.temp_dir / f"{audio_path.stem}_chunks"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Splitting audio into {chunk_length}s chunks with {overlap}s overlap")
        
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=self.audio_config.sample_rate)
            
            # Calculate chunk parameters
            chunk_samples = chunk_length * sr
            overlap_samples = overlap * sr
            step_samples = chunk_samples - overlap_samples
            
            # Split into chunks
            chunks = []
            chunk_paths = []
            
            for i, start in enumerate(range(0, len(audio), step_samples)):
                end = min(start + chunk_samples, len(audio))
                chunk = audio[start:end]
                
                # Skip if chunk is too short
                if len(chunk) < sr:  # Less than 1 second
                    continue
                
                # Save chunk
                chunk_path = output_dir / f"{audio_path.stem}_chunk_{i:04d}.wav"
                sf.write(chunk_path, chunk, sr)
                
                chunks.append(chunk)
                chunk_paths.append(chunk_path)
            
            logger.success(f"Audio split into {len(chunk_paths)} chunks")
            return chunk_paths
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to split audio: {str(e)}",
                file_path=str(audio_path),
                stage="audio_splitting"
            )
    
    def apply_voice_activity_detection(
        self,
        audio_path: Path,
        min_silence_len: int = 500,
        silence_thresh: int = -40,
        output_path: Optional[Path] = None
    ) -> Tuple[Path, List[Tuple[int, int]]]:
        """
        Apply Voice Activity Detection to remove silence
        
        Args:
            audio_path: Path to audio file
            min_silence_len: Minimum silence length in ms
            silence_thresh: Silence threshold in dB
            output_path: Output file path
            
        Returns:
            Tuple of (processed audio path, non-silent segments)
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise AudioProcessingError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Applying VAD to: {audio_path.name}")
        
        try:
            # Load audio
            audio = AudioSegment.from_file(str(audio_path))
            
            # Detect non-silent segments
            non_silent_segments = detect_nonsilent(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh
            )
            
            if not non_silent_segments:
                logger.warning("No speech detected in audio")
                return audio_path, []
            
            # Concatenate non-silent segments
            processed_audio = AudioSegment.empty()
            for start, end in non_silent_segments:
                processed_audio += audio[start:end]
            
            # Determine output path
            if output_path is None:
                output_path = self.temp_dir / f"{audio_path.stem}_vad.wav"
            else:
                output_path = Path(output_path)
            
            # Export processed audio
            processed_audio.export(
                str(output_path),
                format="wav",
                parameters=["-ar", str(self.audio_config.sample_rate)]
            )
            
            # Calculate reduction percentage
            original_duration = len(audio) / 1000  # ms to seconds
            processed_duration = len(processed_audio) / 1000
            reduction = (1 - processed_duration / original_duration) * 100
            
            logger.success(
                f"VAD applied: {original_duration:.1f}s -> {processed_duration:.1f}s "
                f"({reduction:.1f}% silence removed)"
            )
            
            return output_path, non_silent_segments
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to apply VAD: {str(e)}",
                file_path=str(audio_path),
                stage="vad_processing"
            )
    
    def normalize_audio(
        self,
        audio_path: Path,
        target_dBFS: float = -20.0,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Normalize audio volume
        
        Args:
            audio_path: Path to audio file
            target_dBFS: Target volume in dBFS
            output_path: Output file path
            
        Returns:
            Path to normalized audio file
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise AudioProcessingError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Normalizing audio: {audio_path.name}")
        
        try:
            # Load audio
            audio = AudioSegment.from_file(str(audio_path))
            
            # Calculate normalization
            change_in_dBFS = target_dBFS - audio.dBFS
            
            # Apply normalization
            normalized_audio = audio.apply_gain(change_in_dBFS)
            
            # Determine output path
            if output_path is None:
                output_path = self.temp_dir / f"{audio_path.stem}_normalized.wav"
            else:
                output_path = Path(output_path)
            
            # Export
            normalized_audio.export(
                str(output_path),
                format="wav",
                parameters=["-ar", str(self.audio_config.sample_rate)]
            )
            
            logger.success(
                f"Audio normalized: {audio.dBFS:.1f} dBFS -> {target_dBFS:.1f} dBFS"
            )
            
            return output_path
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to normalize audio: {str(e)}",
                file_path=str(audio_path),
                stage="normalization"
            )
    
    def get_audio_info(self, audio_path: Path) -> AudioInfo:
        """
        Get audio file information
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            AudioInfo object
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise AudioProcessingError(f"Audio file not found: {audio_path}")
        
        try:
            # Get file info using ffprobe
            probe = ffmpeg.probe(str(audio_path))
            
            # Extract audio stream info
            audio_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
                None
            )
            
            if not audio_stream:
                raise AudioProcessingError("No audio stream found in file")
            
            # Get duration
            duration = float(probe['format'].get('duration', 0))
            
            # Create AudioInfo object
            info = AudioInfo(
                file_path=audio_path,
                duration=duration,
                sample_rate=int(audio_stream.get('sample_rate', 0)),
                channels=int(audio_stream.get('channels', 0)),
                bit_rate=int(audio_stream.get('bit_rate', 0)) if 'bit_rate' in audio_stream else None,
                codec=audio_stream.get('codec_name'),
                format=probe['format'].get('format_name', audio_path.suffix[1:]),
                file_size=os.path.getsize(audio_path)
            )
            
            return info
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise AudioProcessingError(
                f"Failed to get audio info: {error_msg}",
                file_path=str(audio_path),
                stage="info_extraction"
            )
    
    def validate_audio_file(self, audio_path: Path) -> bool:
        """
        Validate audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if invalid
        """
        audio_path = Path(audio_path)
        
        # Check existence
        if not audio_path.exists():
            raise ValidationError(f"File not found: {audio_path}")
        
        # Check format
        file_format = audio_path.suffix[1:].lower()
        if file_format not in self.audio_config.allowed_formats:
            raise ValidationError(
                f"Unsupported format: {file_format}",
                field="format",
                value=file_format,
                details={"allowed_formats": self.audio_config.allowed_formats}
            )
        
        # Check file size
        file_size = os.path.getsize(audio_path)
        if file_size > self.audio_config.max_file_size:
            raise ValidationError(
                f"File too large: {file_size / (1024**3):.2f} GB",
                field="file_size",
                value=file_size,
                details={
                    "max_size": self.audio_config.max_file_size,
                    "max_size_gb": self.audio_config.max_file_size / (1024**3)
                }
            )
        
        # Get and validate audio info
        try:
            info = self.get_audio_info(audio_path)
            
            # Check duration
            max_duration = 3 * 3600  # 3 hours
            if info.duration > max_duration:
                raise ValidationError(
                    f"Audio too long: {info.duration_minutes:.1f} minutes",
                    field="duration",
                    value=info.duration,
                    details={"max_duration_hours": max_duration / 3600}
                )
            
            logger.success(f"Audio file validated: {audio_path.name}")
            return True
            
        except AudioProcessingError as e:
            raise ValidationError(f"Invalid audio file: {str(e)}")
    
    def process_for_transcription(
        self,
        input_path: Path,
        apply_vad: bool = True,
        normalize: bool = True,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Process audio file for optimal transcription
        
        Args:
            input_path: Input audio/video file
            apply_vad: Apply voice activity detection
            normalize: Normalize audio volume
            output_dir: Output directory
            
        Returns:
            Path to processed audio file
        """
        input_path = Path(input_path)
        
        # Validate input
        self.validate_audio_file(input_path)
        
        # Create output directory
        if output_dir is None:
            output_dir = config.storage.processed_path / input_path.stem
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing audio for transcription: {input_path.name}")
        
        # Check if video file
        video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'webm']
        if input_path.suffix[1:].lower() in video_extensions:
            # Extract audio from video
            audio_path = self.extract_audio_from_video(
                input_path,
                output_path=output_dir / f"{input_path.stem}_extracted.wav"
            )
        else:
            # Convert to WAV if needed
            audio_path = self.convert_audio_format(
                input_path,
                output_format="wav",
                output_path=output_dir / f"{input_path.stem}.wav"
            )
        
        # Apply VAD if requested
        if apply_vad and config.features.vad_enabled:
            audio_path, _ = self.apply_voice_activity_detection(
                audio_path,
                output_path=output_dir / f"{input_path.stem}_vad.wav"
            )
        
        # Normalize if requested
        if normalize:
            audio_path = self.normalize_audio(
                audio_path,
                output_path=output_dir / f"{input_path.stem}_final.wav"
            )
        
        logger.success(f"Audio processed: {audio_path.name}")
        return audio_path
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to clean temp files: {str(e)}")


if __name__ == "__main__":
    # Test audio processor
    logger.print_banner()
    
    # Initialize processor
    processor = AudioProcessor()
    
    # Test with a sample file if exists
    test_file = Path("test_video.mp4")
    if not test_file.exists():
        test_file = Path("test_audio.wav")
    
    if test_file.exists():
        # Get info
        info = processor.get_audio_info(test_file)
        logger.info(f"File: {info.file_path.name}")
        logger.info(f"Duration: {info.duration_minutes:.2f} minutes")
        logger.info(f"Sample Rate: {info.sample_rate} Hz")
        logger.info(f"Channels: {info.channels}")
        logger.info(f"Size: {info.file_size_mb:.2f} MB")
        
        # Process for transcription
        processed = processor.process_for_transcription(test_file)
        logger.success(f"Processed audio: {processed}")
        
        # Cleanup
        processor.cleanup_temp_files()
    else:
        logger.warning("No test file found. Create 'test_video.mp4' or 'test_audio.wav' to test.")