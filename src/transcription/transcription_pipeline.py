"""
Complete transcription pipeline combining audio processing and Whisper
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core import logger
from src.core.config import config
from src.core.exceptions import TranscriptionError, ValidationError
from .whisper_engine import WhisperEngine, WhisperResult, WhisperSegment
from .audio_processor import AudioProcessor, AudioInfo


class TranscriptionPipeline:
    """Complete transcription pipeline"""
    
    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        """
        Initialize transcription pipeline
        
        Args:
            model_size: Whisper model size
            device: Device to use (cuda/cpu/auto)
            batch_size: Batch processing size
        """
        self.model_size = model_size or config.whisper.model_size
        self.device = device or config.whisper.device
        self.batch_size = batch_size or config.processing.batch_size
        
        # Initialize components
        self.audio_processor = AudioProcessor()
        self.whisper_engine = WhisperEngine(
            model_size=self.model_size,
            device=self.device
        )
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "total_duration": 0.0,
            "total_processing_time": 0.0,
            "errors": 0
        }
        
        logger.info("TranscriptionPipeline initialized")
        logger.info(f"Model: {self.model_size} | Device: {self.device} | Batch Size: {self.batch_size}")
    
    def process_file(
        self,
        input_path: Path,
        output_dir: Optional[Path] = None,
        language: Optional[str] = None,
        apply_vad: bool = True,
        normalize_audio: bool = True,
        save_intermediate: bool = False,
        progress_callback=None,
        **transcribe_kwargs
    ) -> Dict[str, Any]:
        """
        Process single file through complete pipeline
        
        Args:
            input_path: Input audio/video file
            output_dir: Output directory for results
            language: Language code for transcription
            apply_vad: Apply voice activity detection
            normalize_audio: Normalize audio volume
            save_intermediate: Save intermediate processed files
            **transcribe_kwargs: Additional transcription parameters
            
        Returns:
            Dictionary with results and metadata
        """
        input_path = Path(input_path)
        start_time = time.time()
        
        # Setup output directory
        if output_dir is None:
            output_dir = config.storage.transcripts_path / input_path.stem
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing file: {input_path.name}")
        
        try:
            # Step 1: Validate input file
            if progress_callback:
                progress_callback("Audio Validation", 10, "Validating audio file...")
            self.audio_processor.validate_audio_file(input_path)
            
            # Step 2: Get original file info
            if progress_callback:
                progress_callback("File Analysis", 15, "Analyzing file properties...")
            original_info = self.audio_processor.get_audio_info(input_path)
            
            # Log file info
            logger.log_transcription_start(
                str(input_path),
                {
                    "Duration": f"{original_info.duration_minutes:.2f} minutes",
                    "Format": original_info.format,
                    "Sample Rate": f"{original_info.sample_rate} Hz",
                    "Size": f"{original_info.file_size_mb:.2f} MB"
                }
            )
            
            # Step 3: Process audio for transcription
            if progress_callback:
                progress_callback("Audio Processing", 25, "Processing audio (VAD, normalization)...")
            processed_audio = self.audio_processor.process_for_transcription(
                input_path,
                apply_vad=apply_vad,
                normalize=normalize_audio,
                output_dir=output_dir if save_intermediate else None
            )
            
            # Step 4: Auto-detect language if not specified
            if progress_callback:
                progress_callback("Language Detection", 60, "Detecting language...")
            if not language and config.features.auto_language_detection:
                detected_lang, confidence = self.whisper_engine.detect_language(processed_audio)
                if confidence > 0.8:
                    language = detected_lang
                    logger.info(f"Auto-detected language: {language} ({confidence:.2%})")
                else:
                    language = config.whisper.language
                    logger.warning(f"Low confidence in language detection, using default: {language}")
            else:
                language = language or config.whisper.language
            
            # Step 5: Transcribe audio
            if progress_callback:
                progress_callback("Transcribing", 65, "Running Whisper transcription...")
            result = self.whisper_engine.transcribe(
                processed_audio,
                language=language,
                **transcribe_kwargs
            )
            
            # Step 6: Save results
            if progress_callback:
                progress_callback("Saving Results", 90, "Saving transcription results...")
            json_path, txt_path, srt_path = result.save(output_dir / input_path.stem)
            
            # Step 7: Create metadata
            processing_time = time.time() - start_time
            metadata = {
                "input_file": str(input_path),
                "output_dir": str(output_dir),
                "original_duration": original_info.duration,
                "original_size_mb": original_info.file_size_mb,
                "processed_audio": str(processed_audio),
                "language": result.language,
                "model": result.model_size,
                "device": result.device,
                "processing_time": processing_time,
                "real_time_factor": processing_time / original_info.duration,
                "word_count": len(result.text.split()),
                "character_count": len(result.text),
                "segment_count": len(result.segments),
                "timestamp": datetime.now().isoformat(),
                "settings": {
                    "vad_applied": apply_vad,
                    "audio_normalized": normalize_audio,
                    "language": language,
                    "transcribe_params": transcribe_kwargs
                },
                "output_files": {
                    "json": str(json_path),
                    "text": str(txt_path),
                    "srt": str(srt_path)
                }
            }
            
            # Save metadata
            metadata_path = output_dir / f"{input_path.stem}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Update statistics
            self.stats["total_processed"] += 1
            self.stats["total_duration"] += original_info.duration
            self.stats["total_processing_time"] += processing_time
            
            # Cleanup temp files if not saving intermediate
            if not save_intermediate:
                self.audio_processor.cleanup_temp_files()
            
            logger.success(f"File processed successfully: {input_path.name}")
            
            return {
                "success": True,
                "result": result,
                "metadata": metadata,
                "error": None
            }
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to process {input_path.name}: {str(e)}")
            
            return {
                "success": False,
                "result": None,
                "metadata": None,
                "error": str(e)
            }
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        recursive: bool = False,
        file_pattern: str = "*",
        parallel: bool = True,
        **process_kwargs
    ) -> Dict[str, Any]:
        """
        Process all files in directory
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            recursive: Process subdirectories
            file_pattern: File pattern to match
            parallel: Process files in parallel
            **process_kwargs: Parameters for process_file
            
        Returns:
            Summary of processing results
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise ValidationError(f"Directory not found: {input_dir}")
        
        # Find files
        if recursive:
            files = list(input_dir.rglob(file_pattern))
        else:
            files = list(input_dir.glob(file_pattern))
        
        # Filter for supported formats
        supported_extensions = config.audio.allowed_formats
        files = [
            f for f in files 
            if f.is_file() and f.suffix[1:].lower() in supported_extensions
        ]
        
        if not files:
            logger.warning(f"No supported files found in {input_dir}")
            return {"processed": 0, "errors": 0, "files": []}
        
        logger.info(f"Found {len(files)} files to process")
        
        # Setup output directory
        if output_dir is None:
            output_dir = config.storage.transcripts_path / input_dir.name
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files
        results = []
        
        if parallel and len(files) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                futures = {
                    executor.submit(
                        self.process_file,
                        file,
                        output_dir / file.stem,
                        **process_kwargs
                    ): file
                    for file in files
                }
                
                # Create progress bar
                progress = logger.create_progress("Batch Processing", len(files))
                
                for future in as_completed(futures):
                    file = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.update_progress("Batch Processing")
                    except Exception as e:
                        logger.error(f"Failed to process {file.name}: {str(e)}")
                        results.append({
                            "success": False,
                            "file": str(file),
                            "error": str(e)
                        })
                
                logger.complete_progress("Batch Processing")
        else:
            # Sequential processing
            for file in files:
                result = self.process_file(
                    file,
                    output_dir / file.stem,
                    **process_kwargs
                )
                results.append(result)
        
        # Create summary
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        
        summary = {
            "total_files": len(files),
            "processed": successful,
            "errors": failed,
            "output_dir": str(output_dir),
            "statistics": self.get_statistics(),
            "results": results
        }
        
        # Save summary
        summary_path = output_dir / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.success(f"Directory processing complete: {successful}/{len(files)} successful")
        
        return summary
    
    def process_chunks(
        self,
        input_path: Path,
        chunk_length: Optional[int] = None,
        overlap: Optional[int] = None,
        **process_kwargs
    ) -> Dict[str, Any]:
        """
        Process long audio by splitting into chunks
        
        Args:
            input_path: Input audio/video file
            chunk_length: Chunk length in seconds
            overlap: Overlap between chunks
            **process_kwargs: Parameters for transcription
            
        Returns:
            Combined results from all chunks
        """
        input_path = Path(input_path)
        
        logger.info(f"Processing file in chunks: {input_path.name}")
        
        # Process audio first
        processed_audio = self.audio_processor.process_for_transcription(input_path)
        
        # Split into chunks
        chunks = self.audio_processor.split_audio_into_chunks(
            processed_audio,
            chunk_length=chunk_length,
            overlap=overlap
        )
        
        logger.info(f"Audio split into {len(chunks)} chunks")
        
        # Transcribe each chunk
        chunk_results = []
        for i, chunk_path in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}")
            
            result = self.whisper_engine.transcribe(
                chunk_path,
                **process_kwargs
            )
            chunk_results.append(result)
        
        # Combine results
        combined_text = " ".join(r.text for r in chunk_results)
        combined_segments = []
        
        offset = 0.0
        for result in chunk_results:
            for segment in result.segments:
                # Adjust timestamps
                adjusted_segment = WhisperSegment(
                    id=len(combined_segments),
                    start=segment.start + offset,
                    end=segment.end + offset,
                    text=segment.text,
                    tokens=segment.tokens,
                    temperature=segment.temperature,
                    avg_logprob=segment.avg_logprob,
                    compression_ratio=segment.compression_ratio,
                    no_speech_prob=segment.no_speech_prob
                )
                combined_segments.append(adjusted_segment)
            
            # Update offset for next chunk
            if result.segments:
                offset = combined_segments[-1].end
        
        # Create combined result
        combined_result = WhisperResult(
            text=combined_text,
            segments=combined_segments,
            language=chunk_results[0].language if chunk_results else "tr",
            duration=combined_segments[-1].end if combined_segments else 0.0,
            processing_time=sum(r.processing_time for r in chunk_results),
            model_size=self.model_size,
            device=self.device,
            audio_file=str(input_path),
            timestamp=datetime.now().isoformat()
        )
        
        # Cleanup chunk files
        for chunk_path in chunks:
            chunk_path.unlink()
        
        return {
            "result": combined_result,
            "chunks_processed": len(chunks),
            "chunk_length": chunk_length or config.audio.chunk_length,
            "overlap": overlap or config.audio.overlap
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        avg_rtf = (
            self.stats["total_processing_time"] / self.stats["total_duration"]
            if self.stats["total_duration"] > 0 else 0
        )
        
        return {
            "total_files_processed": self.stats["total_processed"],
            "total_duration_hours": self.stats["total_duration"] / 3600,
            "total_processing_time_hours": self.stats["total_processing_time"] / 3600,
            "average_real_time_factor": avg_rtf,
            "errors": self.stats["errors"],
            "model": self.model_size,
            "device": self.device
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.audio_processor.cleanup_temp_files()
        self.whisper_engine.unload_model()
        logger.info("Pipeline resources cleaned up")


if __name__ == "__main__":
    # Test transcription pipeline
    logger.print_banner()
    
    # Initialize pipeline
    pipeline = TranscriptionPipeline(
        model_size="base",
        device="cpu"
    )
    
    # Test with a sample file if exists
    test_file = Path("test_video.mp4")
    if not test_file.exists():
        test_file = Path("test_audio.wav")
    
    if test_file.exists():
        # Process single file
        result = pipeline.process_file(
            test_file,
            apply_vad=True,
            normalize_audio=True,
            save_intermediate=True
        )
        
        if result["success"]:
            logger.success("Pipeline test successful!")
            logger.info(f"Transcription: {result['result'].text[:200]}...")
            logger.log_json(result["metadata"], "Processing Metadata")
        else:
            logger.error(f"Pipeline test failed: {result['error']}")
        
        # Get statistics
        stats = pipeline.get_statistics()
        logger.log_json(stats, "Pipeline Statistics")
        
        # Cleanup
        pipeline.cleanup()
    else:
        logger.warning("No test file found. Create 'test_video.mp4' or 'test_audio.wav' to test.")