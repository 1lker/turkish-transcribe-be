"""
YouTube download API endpoints for the FastAPI application.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl, validator
from typing import Optional, Dict, Any, List
import asyncio
import json
from datetime import datetime

from ..core.config import get_config
from ..core.logger import get_logger
from ..core.exceptions import YouTubeDownloadError, InvalidURLError
from ..ingestion.youtube_downloader import create_youtube_downloader
from .models import (
    YouTubeInfoResponse, 
    YouTubeDownloadRequest, 
    YouTubeDownloadResponse,
    DownloadProgressUpdate,
    YouTubeFormatsResponse
)

logger = get_logger(__name__)
config = get_config()
router = APIRouter(prefix="/youtube", tags=["youtube"])

# Initialize YouTube downloader
youtube_downloader = create_youtube_downloader(config)

# Active downloads tracking
active_downloads: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, WebSocket] = {}


class YouTubeURLRequest(BaseModel):
    """Request model for YouTube URL operations."""
    url: HttpUrl
    
    @validator('url')
    def validate_youtube_url(cls, v):
        """Validate that the URL is a YouTube URL."""
        url_str = str(v)
        if not any(domain in url_str for domain in ['youtube.com', 'youtu.be']):
            raise ValueError('URL must be a YouTube URL')
        return v


class YouTubeDownloadOptionsRequest(BaseModel):
    """Request model for YouTube download with options."""
    url: HttpUrl
    quality: Optional[str] = 'best'
    session_id: Optional[str] = None
    
    @validator('url')
    def validate_youtube_url(cls, v):
        url_str = str(v)
        if not any(domain in url_str for domain in ['youtube.com', 'youtu.be']):
            raise ValueError('URL must be a YouTube URL')
        return v
    
    @validator('quality')
    def validate_quality(cls, v):
        if v not in ['best', 'high', 'medium']:
            raise ValueError('Quality must be one of: best, high, medium')
        return v


@router.get("/info", response_model=YouTubeInfoResponse)
async def get_youtube_info(url: str):
    """
    Get information about a YouTube video without downloading.
    
    Args:
        url: YouTube video URL
        
    Returns:
        Video information including title, duration, and metadata
    """
    try:
        logger.info(f"Getting info for YouTube URL: {url}")
        
        # Validate URL
        request = YouTubeURLRequest(url=url)
        
        # Get video info
        info = await youtube_downloader.get_video_info(str(request.url))
        
        return YouTubeInfoResponse(
            success=True,
            video_id=info['id'],
            title=info['title'],
            duration=info['duration'],
            description=info['description'],
            uploader=info['uploader'],
            upload_date=info['upload_date'],
            view_count=info['view_count'],
            like_count=info['like_count'],
            thumbnail=info['thumbnail'],
            has_audio=info['has_audio'],
            formats_count=info['formats_available']
        )
        
    except InvalidURLError as e:
        logger.error(f"Invalid URL provided: {url}")
        raise HTTPException(status_code=400, detail=str(e))
    except YouTubeDownloadError as e:
        logger.error(f"YouTube error for URL {url}: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting info for {url}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get video information")


@router.get("/formats", response_model=YouTubeFormatsResponse)
async def get_youtube_formats(url: str):
    """
    Get available audio formats for a YouTube video.
    
    Args:
        url: YouTube video URL
        
    Returns:
        List of available audio formats
    """
    try:
        logger.info(f"Getting formats for YouTube URL: {url}")
        
        # Validate URL
        request = YouTubeURLRequest(url=url)
        
        # Get available formats
        formats = await youtube_downloader.get_available_formats(str(request.url))
        
        return YouTubeFormatsResponse(
            success=True,
            formats=formats,
            total_formats=len(formats)
        )
        
    except InvalidURLError as e:
        logger.error(f"Invalid URL provided: {url}")
        raise HTTPException(status_code=400, detail=str(e))
    except YouTubeDownloadError as e:
        logger.error(f"YouTube error for URL {url}: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting formats for {url}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get video formats")


@router.post("/download", response_model=YouTubeDownloadResponse)
async def download_youtube_audio(
    request: YouTubeDownloadOptionsRequest,
    background_tasks: BackgroundTasks
):
    """
    Download audio from a YouTube video.
    
    Args:
        request: Download request with URL and options
        background_tasks: FastAPI background tasks
        
    Returns:
        Download initiation response with session ID
    """
    try:
        logger.info(f"Starting download for YouTube URL: {request.url}")
        
        # Generate session ID if not provided
        session_id = request.session_id or f"yt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check if already downloading
        if session_id in active_downloads:
            raise HTTPException(
                status_code=409, 
                detail=f"Download already in progress for session {session_id}"
            )
        
        # Initialize download tracking
        active_downloads[session_id] = {
            'status': 'initializing',
            'url': str(request.url),
            'quality': request.quality,
            'started_at': datetime.now().isoformat(),
            'progress': 0
        }
        
        # Start download in background
        background_tasks.add_task(
            _download_youtube_audio,
            session_id,
            str(request.url),
            request.quality
        )
        
        return YouTubeDownloadResponse(
            success=True,
            session_id=session_id,
            status='started',
            message=f"Download started for session {session_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start download for {request.url}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start download")


@router.get("/download/{session_id}/status")
async def get_download_status(session_id: str):
    """
    Get the status of a download session.
    
    Args:
        session_id: Download session ID
        
    Returns:
        Current download status and progress
    """
    if session_id not in active_downloads:
        raise HTTPException(status_code=404, detail="Download session not found")
    
    return active_downloads[session_id]


@router.delete("/download/{session_id}")
async def cancel_download(session_id: str):
    """
    Cancel an active download.
    
    Args:
        session_id: Download session ID
        
    Returns:
        Cancellation confirmation
    """
    if session_id not in active_downloads:
        raise HTTPException(status_code=404, detail="Download session not found")
    
    # Mark as cancelled
    active_downloads[session_id]['status'] = 'cancelled'
    active_downloads[session_id]['cancelled_at'] = datetime.now().isoformat()
    
    # Notify via WebSocket if connected
    if session_id in websocket_connections:
        try:
            await websocket_connections[session_id].send_text(json.dumps({
                'type': 'status_update',
                'status': 'cancelled',
                'message': 'Download cancelled by user'
            }))
        except:
            pass
    
    return {"success": True, "message": f"Download {session_id} cancelled"}


@router.websocket("/download/{session_id}/ws")
async def download_progress_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time download progress updates.
    
    Args:
        websocket: WebSocket connection
        session_id: Download session ID
    """
    await websocket.accept()
    websocket_connections[session_id] = websocket
    
    try:
        # Send initial status if download exists
        if session_id in active_downloads:
            await websocket.send_text(json.dumps({
                'type': 'status_update',
                'session_id': session_id,
                **active_downloads[session_id]
            }))
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for client messages (ping/pong or commands)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)
                
                if message.get('type') == 'ping':
                    await websocket.send_text(json.dumps({'type': 'pong'}))
                    
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_text(json.dumps({'type': 'ping'}))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {str(e)}")
    finally:
        # Clean up connection
        if session_id in websocket_connections:
            del websocket_connections[session_id]


async def _download_youtube_audio(session_id: str, url: str, quality: str):
    """
    Background task for downloading YouTube audio.
    
    Args:
        session_id: Download session ID
        url: YouTube URL
        quality: Audio quality setting
    """
    try:
        logger.info(f"Background download started for session {session_id}")
        
        # Update status
        active_downloads[session_id]['status'] = 'downloading'
        await _send_websocket_update(session_id, {
            'type': 'status_update',
            'status': 'downloading',
            'message': 'Download started'
        })
        
        # Progress callback for real-time updates
        async def progress_callback(progress_data):
            if session_id in active_downloads:
                # Check if cancelled
                if active_downloads[session_id].get('status') == 'cancelled':
                    raise YouTubeDownloadError("Download cancelled by user")
                
                # Update progress
                active_downloads[session_id]['progress'] = progress_data.get('percent', '0%')
                active_downloads[session_id]['speed'] = progress_data.get('speed', 'N/A')
                active_downloads[session_id]['eta'] = progress_data.get('eta', 'N/A')
                
                # Send WebSocket update
                await _send_websocket_update(session_id, {
                    'type': 'progress_update',
                    **progress_data
                })
        
        # Download the audio
        result = await youtube_downloader.download_audio(
            url=url,
            quality=quality,
            progress_callback=progress_callback
        )
        
        # Update with success
        active_downloads[session_id].update({
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'result': result
        })
        
        await _send_websocket_update(session_id, {
            'type': 'status_update',
            'status': 'completed',
            'result': result,
            'message': 'Download completed successfully'
        })
        
        logger.info(f"Download completed for session {session_id}: {result['filename']}")
        
    except YouTubeDownloadError as e:
        error_msg = str(e)
        logger.error(f"YouTube download error for session {session_id}: {error_msg}")
        
        active_downloads[session_id].update({
            'status': 'failed',
            'failed_at': datetime.now().isoformat(),
            'error': error_msg
        })
        
        await _send_websocket_update(session_id, {
            'type': 'status_update',
            'status': 'failed',
            'error': error_msg,
            'message': f'Download failed: {error_msg}'
        })
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error for session {session_id}: {error_msg}")
        
        active_downloads[session_id].update({
            'status': 'failed',
            'failed_at': datetime.now().isoformat(),
            'error': error_msg
        })
        
        await _send_websocket_update(session_id, {
            'type': 'status_update',
            'status': 'failed',
            'error': error_msg,
            'message': f'Download failed: {error_msg}'
        })


async def _send_websocket_update(session_id: str, data: Dict[str, Any]):
    """
    Send update to WebSocket connection if exists.
    
    Args:
        session_id: Session ID
        data: Data to send
    """
    if session_id in websocket_connections:
        try:
            await websocket_connections[session_id].send_text(json.dumps({
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                **data
            }))
        except Exception as e:
            logger.error(f"Failed to send WebSocket update for {session_id}: {str(e)}")
            # Remove broken connection
            if session_id in websocket_connections:
                del websocket_connections[session_id]


# Background task for cleanup
async def cleanup_old_downloads():
    """Clean up old download sessions and files."""
    try:
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session_data in active_downloads.items():
            if session_data.get('status') in ['completed', 'failed', 'cancelled']:
                # Remove sessions older than 1 hour
                started_at = datetime.fromisoformat(session_data['started_at'])
                if (current_time - started_at).total_seconds() > 3600:
                    sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del active_downloads[session_id]
            logger.info(f"Cleaned up old session: {session_id}")
        
        # Clean up old files
        await youtube_downloader.cleanup_old_downloads(max_age_hours=24)
        
    except Exception as e:
        logger.error(f"Cleanup task failed: {str(e)}")


# Add cleanup task to router startup
@router.on_event("startup")
async def setup_cleanup():
    """Setup periodic cleanup task."""
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(3600)  # Run every hour
            await cleanup_old_downloads()
    
    asyncio.create_task(periodic_cleanup())
