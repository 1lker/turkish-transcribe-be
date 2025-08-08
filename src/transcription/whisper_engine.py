"""
Whisper Engine wrapper for transcription
"""

import os
import json
import time
import torch
import whisper
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from src.core import logger
from src.core.config import config
from src.core.exceptions import (
    ModelLoadError,
    TranscriptionError,
    AudioProcessingError,
    GPUError
)

# Suppress FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")


@dataclass
class WhisperSegment:
    """Single transcription segment"""
    id: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    
    @property
    def duration(self) -> float:
        """Get segment duration"""
        return self.end - self.start
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class WhisperResult:
    """Transcription result container"""
    text: str
    segments: List[WhisperSegment]
    language: str
    duration: float
    processing_time: float
    model_size: str
    device: str
    audio_file: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "segments": [seg.to_dict() for seg in self.segments],
            "language": self.language,
            "duration": self.duration,
            "processing_time": self.processing_time,
            "model_size": self.model_size,
            "device": self.device,
            "audio_file": self.audio_file,
            "timestamp": self.timestamp
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def save(self, output_path: Path):
        """Save result to file"""
        output_path = Path(output_path)
        
        # Save as JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        
        # Save as plain text
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(self.text)
        
        # Save as SRT
        srt_path = output_path.with_suffix('.srt')
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(self.to_srt())
        
        logger.success(f"Results saved to: {output_path.parent}")
        return json_path, txt_path, srt_path
    
    def to_srt(self) -> str:
        """Convert to SRT subtitle format"""
        srt_lines = []
        for i, segment in enumerate(self.segments, 1):
            start = self._seconds_to_srt_time(segment.start)
            end = self._seconds_to_srt_time(segment.end)
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(segment.text.strip())
            srt_lines.append("")
        return "\n".join(srt_lines)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class WhisperEngine:
    """Whisper transcription engine"""
    
    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize Whisper engine
        
        Args:
            model_size: Model size (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to use (cuda, cpu, auto)
        """
        self.model_size = model_size or config.whisper.model_size
        self.device = self._setup_device(device or config.whisper.device)
        self.model = None
        self.whisper_config = config.whisper
        
        logger.info(f"Initializing Whisper Engine")
        logger.info(f"Model: {self.model_size} | Device: {self.device}")
        
        # Load model
        self.load_model()
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        if device == "cuda":
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        return device
    
    def load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            start_time = time.time()
            
            # Download and load model
            self.model = whisper.load_model(
                name=self.model_size,
                device=self.device,
                download_root=Path("models/whisper")
            )
            
            load_time = time.time() - start_time
            logger.success(f"Model loaded successfully in {load_time:.2f} seconds")
            
            # Model info
            n_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model parameters: {n_params / 1e6:.2f}M")
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load Whisper model: {str(e)}",
                model_name="whisper",
                model_size=self.model_size
            )
    
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
        **kwargs
    ) -> WhisperResult:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code (tr, en, etc.)
            initial_prompt: Initial prompt for context
            task: Task type (transcribe or translate)
            verbose: Show progress
            **kwargs: Additional whisper parameters
            
        Returns:
            WhisperResult object
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise AudioProcessingError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Starting transcription: {audio_path.name}")
        start_time = time.time()
        
        try:
            # Prepare parameters
            transcribe_params = self._prepare_transcribe_params(
                language=language,
                initial_prompt=initial_prompt,
                task=task,
                verbose=verbose,
                **kwargs
            )
            
            # Log parameters
            logger.debug(f"Transcription parameters: {transcribe_params}")
            
            # Transcribe
            result = self.model.transcribe(
                str(audio_path),
                **transcribe_params
            )
            
            # Process result
            processing_time = time.time() - start_time
            whisper_result = self._process_result(
                result=result,
                audio_path=audio_path,
                processing_time=processing_time
            )
            
            # Log statistics
            self._log_statistics(whisper_result)
            
            return whisper_result
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise TranscriptionError(
                f"Transcription failed for {audio_path.name}: {str(e)}",
                details={"audio_file": str(audio_path), "error": str(e)}
            )
    
    def _prepare_transcribe_params(
        self,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare transcription parameters"""
        params = {
            "language": language or self.whisper_config.language,
            "task": task,
            "verbose": verbose,
            "temperature": kwargs.get("temperature", self.whisper_config.temperature),
            "compression_ratio_threshold": kwargs.get(
                "compression_ratio_threshold",
                self.whisper_config.compression_ratio_threshold
            ),
            "logprob_threshold": kwargs.get(
                "logprob_threshold",
                self.whisper_config.logprob_threshold
            ),
            "no_speech_threshold": kwargs.get(
                "no_speech_threshold",
                self.whisper_config.no_speech_threshold
            ),
            "condition_on_previous_text": kwargs.get(
                "condition_on_previous_text",
                self.whisper_config.condition_on_previous_text
            ),
            "fp16": self.device == "cuda",
            "beam_size": kwargs.get("beam_size", self.whisper_config.beam_size),
            "patience": kwargs.get("patience", self.whisper_config.patience),
            "length_penalty": kwargs.get("length_penalty", self.whisper_config.length_penalty),
            "suppress_tokens": kwargs.get("suppress_tokens", self.whisper_config.suppress_tokens),
        }
        
        # Add initial prompt if provided
        if initial_prompt or self.whisper_config.initial_prompt:
            params["initial_prompt"] = initial_prompt or self.whisper_config.initial_prompt
        
        # Add best_of for non-zero temperature
        if params["temperature"] > 0:
            params["best_of"] = kwargs.get("best_of", self.whisper_config.best_of)
        
        return params
    
    def _process_result(
        self,
        result: Dict[str, Any],
        audio_path: Path,
        processing_time: float
    ) -> WhisperResult:
        """Process Whisper result into WhisperResult object"""
        
        # Extract segments
        segments = []
        for seg in result.get("segments", []):
            segment = WhisperSegment(
                id=seg["id"],
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                tokens=seg["tokens"],
                temperature=seg["temperature"],
                avg_logprob=seg["avg_logprob"],
                compression_ratio=seg["compression_ratio"],
                no_speech_prob=seg["no_speech_prob"]
            )
            segments.append(segment)
        
        # Calculate total duration
        duration = segments[-1].end if segments else 0.0
        
        # Create result object
        whisper_result = WhisperResult(
            text=result["text"],
            segments=segments,
            language=result.get("language", self.whisper_config.language),
            duration=duration,
            processing_time=processing_time,
            model_size=self.model_size,
            device=self.device,
            audio_file=str(audio_path),
            timestamp=datetime.now().isoformat()
        )
        
        return whisper_result
    
    def _log_statistics(self, result: WhisperResult):
        """Log transcription statistics"""
        stats = {
            "Total Duration": f"{result.duration:.2f} seconds",
            "Processing Time": f"{result.processing_time:.2f} seconds",
            "Real-time Factor": f"{result.processing_time / result.duration:.2f}x",
            "Number of Segments": len(result.segments),
            "Total Words": len(result.text.split()),
            "Characters": len(result.text),
            "Language": result.language,
            "Model": result.model_size,
            "Device": result.device
        }
        
        logger.log_transcription_complete(
            file_path=result.audio_file,
            duration=result.processing_time,
            stats=stats
        )
    
    def batch_transcribe(
        self,
        audio_files: List[Path],
        output_dir: Optional[Path] = None,
        **kwargs
    ) -> List[WhisperResult]:
        """
        Transcribe multiple audio files
        
        Args:
            audio_files: List of audio file paths
            output_dir: Directory to save results
            **kwargs: Transcription parameters
            
        Returns:
            List of WhisperResult objects
        """
        results = []
        output_dir = output_dir or config.storage.transcripts_path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting batch transcription for {len(audio_files)} files")
        
        # Create progress bar
        progress = logger.create_progress("Batch Transcription", len(audio_files))
        
        for i, audio_file in enumerate(audio_files, 1):
            try:
                logger.info(f"Processing file {i}/{len(audio_files)}: {audio_file.name}")
                
                # Transcribe
                result = self.transcribe(audio_file, **kwargs)
                
                # Save result
                output_path = output_dir / audio_file.stem
                result.save(output_path)
                
                results.append(result)
                logger.update_progress("Batch Transcription")
                
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_file.name}: {str(e)}")
                continue
        
        logger.complete_progress("Batch Transcription")
        logger.success(f"Batch transcription completed: {len(results)}/{len(audio_files)} successful")
        
        return results
    
    def detect_language(self, audio_path: Path) -> Tuple[str, float]:
        """
        Detect language of audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (language_code, probability)
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise AudioProcessingError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Detecting language for: {audio_path.name}")
        
        # Load audio and detect language
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)
        
        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        
        # Detect language
        _, probs = self.model.detect_language(mel)
        
        # Get top language
        language = max(probs, key=probs.get)
        probability = probs[language]
        
        logger.info(f"Detected language: {language} (confidence: {probability:.2%})")
        
        return language, probability
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.model:
            return {"error": "Model not loaded"}
        
        n_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "model_size": self.model_size,
            "device": self.device,
            "parameters": f"{n_params / 1e6:.2f}M",
            "dims": {
                "n_mels": self.model.dims.n_mels,
                "n_audio_ctx": self.model.dims.n_audio_ctx,
                "n_audio_state": self.model.dims.n_audio_state,
                "n_audio_head": self.model.dims.n_audio_head,
                "n_audio_layer": self.model.dims.n_audio_layer,
                "n_text_ctx": self.model.dims.n_text_ctx,
                "n_text_state": self.model.dims.n_text_state,
                "n_text_head": self.model.dims.n_text_head,
                "n_text_layer": self.model.dims.n_text_layer,
            },
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    
    def unload_model(self):
        """Unload model from memory"""
        if self.model:
            del self.model
            self.model = None
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded from memory")


if __name__ == "__main__":
    # Test Whisper engine
    logger.print_banner()
    
    # Initialize engine
    engine = WhisperEngine(model_size="base", device="cpu")
    
    # Get model info
    info = engine.get_model_info()
    logger.log_json(info, "Model Information")
    
    # Test with a sample audio file if exists
    test_audio = Path("test_audio.wav")
    if test_audio.exists():
        # Detect language
        lang, prob = engine.detect_language(test_audio)
        logger.info(f"Language: {lang} ({prob:.2%})")
        
        # Transcribe
        result = engine.transcribe(test_audio)
        
        # Save results
        output_path = Path("data/transcripts") / test_audio.stem
        result.save(output_path)
        
        logger.success("Test completed!")
    else:
        logger.warning("No test audio file found. Create 'test_audio.wav' to test transcription.")
    
    # Unload model
    engine.unload_model()