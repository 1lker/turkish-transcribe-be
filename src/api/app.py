"""
FastAPI application for Turkish Education Transcription System
"""

import os
import uuid
import torch
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.core import logger
from src.core.config import config
from src.core.exceptions import TranscriptionError, ValidationError
from src.transcription import TranscriptionPipeline, WhisperEngine, AudioProcessor

from .models import (
    TranscriptionRequest,
    TranscriptionResponse,
    FileInfoResponse,
    LanguageDetectionResponse,
    BatchTranscriptionRequest,
    BatchTranscriptionResponse,
    TaskStatusResponse,
    HealthResponse,
    ErrorResponse,
    ProcessingStatus,
    WebSocketMessage,
    YouTubeInfoResponse,
    YouTubeDownloadRequest,
    YouTubeDownloadResponse,
    YouTubeFormatsResponse
)

# Import YouTube routes
from .youtube_routes import router as youtube_router


# Global variables for pipeline and tasks
pipeline: Optional[TranscriptionPipeline] = None
tasks: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.print_banner()
    logger.info("Starting FastAPI application...")
    
    # Initialize pipeline
    global pipeline
    pipeline = TranscriptionPipeline(
        model_size=config.whisper.model_size,
        device=config.whisper.device
    )
    logger.success("Pipeline initialized successfully")
    
    # Create necessary directories
    config.storage.create_directories()
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    if pipeline:
        pipeline.cleanup()
    logger.success("Application shutdown complete")


def get_application() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Turkish Education Transcription API",
        description="🎓 Powered by Whisper AI | 🇹🇷 Optimized for Turkish",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(youtube_router)
    
    return app


# Create application instance
app = get_application()


# ============= Health & Status Endpoints =============

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Turkish Education Transcription API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download a file from the raw data directory.
    
    Args:
        filename: Name of the file to download
        
    Returns:
        File response
    """
    try:
        from pathlib import Path
        
        # Try both possible paths
        file_path = config.storage.raw_videos_path / filename
        
        if not file_path.exists():
            logger.error(f"File not found: {filename} in {config.storage.raw_videos_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        logger.info(f"Serving file: {file_path}")
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='audio/wav'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download file")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(),
        whisper_model_loaded=pipeline is not None,
        available_models=["tiny", "base", "small", "medium", "large"],
        gpu_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )


# ============= File Upload & Info Endpoints =============

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload a file for transcription"""
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    file_ext = Path(file.filename).suffix[1:].lower()
    if file_ext not in config.audio.allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file_ext}. Allowed: {config.audio.allowed_formats}"
        )
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    file_path = config.storage.raw_videos_path / f"{file_id}_{file.filename}"
    
    # Save file
    try:
        content = await file.read()
        
        # Check file size
        if len(content) > config.audio.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {config.audio.max_file_size / (1024**3):.2f} GB"
            )
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"File uploaded: {file.filename} -> {file_id}")
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "size": len(content),
            "path": str(file_path)
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/file/{file_id}/info", response_model=FileInfoResponse)
async def get_file_info(file_id: str):
    """Get information about an uploaded file"""
    # Find file
    file_pattern = f"{file_id}_*"
    files = list(config.storage.raw_videos_path.glob(file_pattern))
    
    if not files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = files[0]
    
    try:
        # Get file info
        processor = AudioProcessor()
        info = processor.get_audio_info(file_path)
        
        return FileInfoResponse(
            filename=file_path.name,
            format=info.format,
            duration=info.duration,
            duration_minutes=info.duration_minutes,
            sample_rate=info.sample_rate,
            channels=info.channels,
            codec=info.codec,
            bit_rate=info.bit_rate,
            file_size=info.file_size,
            file_size_mb=info.file_size_mb
        )
        
    except Exception as e:
        logger.error(f"Failed to get file info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= Transcription Endpoints =============

@app.post("/transcribe/{file_id}", response_model=TranscriptionResponse)
async def transcribe_file(
    file_id: str,
    request: TranscriptionRequest,
    background_tasks: BackgroundTasks
):
    """Start transcription for an uploaded file"""
    # Find file
    file_pattern = f"{file_id}_*"
    files = list(config.storage.raw_videos_path.glob(file_pattern))
    
    if not files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = files[0]
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task
    tasks[task_id] = {
        "status": ProcessingStatus.PENDING,
        "file_path": str(file_path),
        "request": request,
        "created_at": datetime.now(),
        "progress": 0.0
    }
    
    # Start transcription in background
    background_tasks.add_task(
        process_transcription,
        task_id,
        file_path,
        request
    )
    
    logger.info(f"Transcription started: {task_id}")
    
    return TranscriptionResponse(
        task_id=task_id,
        status=ProcessingStatus.PENDING,
        created_at=datetime.now()
    )


async def process_transcription(
    task_id: str,
    file_path: Path,
    request: TranscriptionRequest
):
    """Process transcription in background"""
    try:
        # Update status
        tasks[task_id]["status"] = ProcessingStatus.PROCESSING
        await notify_websocket(task_id, "status", {"status": "processing", "progress": 0})
        
        # Create new pipeline with requested model (HER REQUEST İÇİN YENİ PİPELINE)
        from src.transcription import TranscriptionPipeline
        
        logger.info(f"Creating pipeline with model: {request.model_size}")
        
        # Request'ten gelen model_size ile yeni pipeline oluştur
        custom_pipeline = TranscriptionPipeline(
            model_size=request.model_size or "base",  # Request'ten gelen model
            device=request.device if request.device != "auto" else None
        )
        
        # Progress updates during processing
        await notify_websocket(task_id, "progress", {"progress": 10, "stage": "Initializing", "message": "Setting up pipeline..."})
        
        # Create a queue for progress updates
        import queue
        import threading
        progress_queue = queue.Queue()
        
        def progress_callback(stage: str, progress: float, message: str = ""):
            # Put progress update in queue
            progress_queue.put({
                "progress": progress,
                "stage": stage, 
                "message": message
            })
            print(f"🔄 Progress: {stage} - {progress}% - {message}")  # Debug log
        
        # Function to send queued progress updates
        async def send_progress_updates():
            while not progress_queue.empty():
                try:
                    update = progress_queue.get_nowait()
                    await notify_websocket(task_id, "progress", update)
                    await asyncio.sleep(0.1)  # Small delay
                except queue.Empty:
                    break
        
        # Process file with progress callback in separate thread
        await notify_websocket(task_id, "progress", {"progress": 30, "stage": "Processing audio", "message": "Starting audio processing..."})
        
        def run_transcription():
            return custom_pipeline.process_file(
                input_path=file_path,
                language=request.language,
                apply_vad=request.apply_vad,
                normalize_audio=request.normalize_audio,
                progress_callback=progress_callback
            )
        
        # Run transcription in thread and periodically check for progress
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_transcription)
            
            # Periodically send progress updates
            while not future.done():
                await send_progress_updates()
                await asyncio.sleep(1)  # Check every second
            
            # Get the result
            result = future.result()
            
            # Send any remaining progress updates
            await send_progress_updates()
        
        await notify_websocket(task_id, "progress", {"progress": 90, "stage": "Finalizing", "message": "Saving results..."})
        
        # Cleanup custom pipeline (belleği temizle)
        custom_pipeline.cleanup()
        
        if result["success"]:
            await notify_websocket(task_id, "progress", {"progress": 100, "stage": "Completed"})
            
            # Update task with results
            tasks[task_id].update({
                "status": ProcessingStatus.COMPLETED,
                "result": result["result"],
                "metadata": result["metadata"],
                "completed_at": datetime.now()
            })
            
            # Notify via websocket
            await notify_websocket(task_id, "result", {
                "status": "completed",
                "text": result["result"].text[:500]  # Preview
            })
            
            logger.success(f"Transcription completed: {task_id}")
            
        else:
            tasks[task_id].update({
                "status": ProcessingStatus.FAILED,
                "error": result["error"],
                "completed_at": datetime.now()
            })
            
            await notify_websocket(task_id, "error", {"error": result["error"]})
            logger.error(f"Transcription failed: {task_id}")
            
    except Exception as e:
        tasks[task_id].update({
            "status": ProcessingStatus.FAILED,
            "error": str(e),
            "completed_at": datetime.now()
        })
        
        await notify_websocket(task_id, "error", {"error": str(e)})
        logger.error(f"Transcription error: {str(e)}")


@app.get("/task/{task_id}", response_model=TranscriptionResponse)
async def get_task_status(task_id: str):
    """Get transcription task status"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    response = TranscriptionResponse(
        task_id=task_id,
        status=task["status"],
        created_at=task["created_at"],
        completed_at=task.get("completed_at")
    )
    
    if task["status"] == ProcessingStatus.COMPLETED:
        result = task["result"]
        metadata = task["metadata"]
        
        response.text = result.text
        response.language = result.language
        response.duration = metadata["original_duration"]
        response.processing_time = metadata["processing_time"]
        response.word_count = metadata["word_count"]
        response.output_files = metadata["output_files"]
        
    elif task["status"] == ProcessingStatus.FAILED:
        response.error = task.get("error")
    
    return response


@app.get("/task/{task_id}/download/{format}")
async def download_result(task_id: str, format: str):
    """Download transcription result in specified format"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task["status"] != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task not completed")
    
    # Get output file path
    output_files = task["metadata"]["output_files"]
    
    # Format mapping (txt -> text)
    format_mapping = {
        "txt": "text",
        "json": "json", 
        "srt": "srt"
    }
    
    mapped_format = format_mapping.get(format, format)
    
    if mapped_format not in output_files:
        raise HTTPException(status_code=400, detail=f"Format {format} not available")
    
    file_path = Path(output_files[mapped_format])
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/octet-stream"
    )

# ============= Language Detection Endpoint =============

@app.post("/detect-language/{file_id}", response_model=LanguageDetectionResponse)
async def detect_language(file_id: str):
    """Detect language of an audio file"""
    # Find file
    file_pattern = f"{file_id}_*"
    files = list(config.storage.raw_videos_path.glob(file_pattern))
    
    if not files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = files[0]
    
    try:
        # Detect language
        engine = WhisperEngine(model_size="base")
        language, confidence = engine.detect_language(file_path)
        
        # Language names
        language_names = {
            'tr': 'Turkish',
            'en': 'English',
            'de': 'German',
            'fr': 'French',
            'es': 'Spanish',
            'ar': 'Arabic',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean'
        }
        
        response = LanguageDetectionResponse(
            detected_language=language,
            confidence=confidence,
            language_name=language_names.get(language)
        )
        
        # Cleanup
        engine.unload_model()
        
        return response
        
    except Exception as e:
        logger.error(f"Language detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= Batch Processing Endpoint =============

@app.post("/batch/transcribe", response_model=BatchTranscriptionResponse)
async def batch_transcribe(
    request: BatchTranscriptionRequest,
    background_tasks: BackgroundTasks
):
    """Start batch transcription for multiple files"""
    # Validate files exist
    file_paths = []
    for file_id in request.file_ids:
        file_pattern = f"{file_id}_*"
        files = list(config.storage.raw_videos_path.glob(file_pattern))
        
        if not files:
            raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
        
        file_paths.append(files[0])
    
    # Generate batch ID
    batch_id = f"batch-{uuid.uuid4()}"
    
    # Create batch task
    batch_results = []
    
    for file_path in file_paths:
        task_id = str(uuid.uuid4())
        
        tasks[task_id] = {
            "status": ProcessingStatus.PENDING,
            "file_path": str(file_path),
            "batch_id": batch_id,
            "created_at": datetime.now()
        }
        
        batch_results.append(
            TranscriptionResponse(
                task_id=task_id,
                status=ProcessingStatus.PENDING,
                created_at=datetime.now()
            )
        )
        
        # Add to background tasks
        background_tasks.add_task(
            process_transcription,
            task_id,
            file_path,
            request.settings
        )
    
    return BatchTranscriptionResponse(
        batch_id=batch_id,
        total_files=len(file_paths),
        completed=0,
        failed=0,
        pending=len(file_paths),
        results=batch_results,
        created_at=datetime.now()
    )


# ============= WebSocket Endpoint =============

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    
    # Add connection to pool
    if task_id not in websocket_connections:
        websocket_connections[task_id] = []
    websocket_connections[task_id].append(websocket)
    
    try:
        # Send initial status
        if task_id in tasks:
            await websocket.send_json({
                "type": "status",
                "task_id": task_id,
                "data": {"status": tasks[task_id]["status"].value}
            })
        
        # Keep connection alive
        while True:
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        # Remove connection
        if task_id in websocket_connections:
            websocket_connections[task_id].remove(websocket)
            if not websocket_connections[task_id]:
                del websocket_connections[task_id]


async def notify_websocket(task_id: str, message_type: str, data: Dict[str, Any]):
    """Send notification to WebSocket clients"""
    logger.info(f"🔌 notify_websocket called: task_id={task_id}, type={message_type}, connections={len(websocket_connections.get(task_id, []))}")
    
    if task_id in websocket_connections:
        message = WebSocketMessage(
            type=message_type,
            task_id=task_id,
            data=data
        )
        
        logger.info(f"📤 Sending WebSocket message: {message.model_dump(mode='json')}")
        
        for websocket in websocket_connections[task_id]:
            try:
                await websocket.send_json(message.model_dump(mode='json'))
                logger.info(f"✅ Message sent to WebSocket client")
            except Exception as e:
                logger.warning(f"❌ Failed to send WebSocket message: {e}")
    else:
        logger.warning(f"🔌 No WebSocket connections found for task_id: {task_id}")


# ============= Error Handlers =============

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            status_code=400
        ).model_dump()
    )


@app.exception_handler(TranscriptionError)
async def transcription_exception_handler(request, exc: TranscriptionError):
    """Handle transcription errors"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Transcription Error",
            detail=str(exc),
            status_code=500
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.app:app",
        host=config.server.get("host", "0.0.0.0"),
        port=config.server.get("port", 8000),
        reload=config.server.get("reload", True),
        log_level="info"
    )