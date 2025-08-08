"""
YouTube downloader module for extracting audio from YouTube videos.
Supports various quality options and formats.
"""

import os
import uuid
import asyncio
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, parse_qs
import yt_dlp
from datetime import datetime

from ..core.logger import get_logger
from ..core.config import Config
from ..core.exceptions import YouTubeDownloadError, InvalidURLError

logger = get_logger(__name__)


class YouTubeDownloader:
    """
    YouTube audio downloader with support for various formats and quality options.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.download_dir = Path(config.storage.raw_videos_path)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported audio formats and their configurations
        self.audio_formats = {
            'best': {
                'format': 'bestaudio/best',
                'ext': 'wav',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }]
            },
            'high': {
                'format': 'bestaudio[ext=m4a]/bestaudio',
                'ext': 'wav',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '160',
                }]
            },
            'medium': {
                'format': 'bestaudio[ext=webm]/bestaudio',
                'ext': 'wav',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '128',
                }]
            }
        }
    
    def _is_valid_youtube_url(self, url: str) -> bool:
        """
        Validate if the provided URL is a valid YouTube URL.
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if valid YouTube URL
        """
        try:
            parsed = urlparse(url)
            youtube_domains = ['youtube.com', 'www.youtube.com', 'youtu.be', 'm.youtube.com']
            
            if parsed.netloc in youtube_domains:
                if parsed.netloc == 'youtu.be':
                    return len(parsed.path) > 1
                else:
                    query_params = parse_qs(parsed.query)
                    return 'v' in query_params and len(query_params['v'][0]) == 11
            
            return False
        except Exception:
            return False
    
    def _extract_video_id(self, url: str) -> str:
        """
        Extract video ID from YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            str: Video ID
        """
        try:
            parsed = urlparse(url)
            
            if parsed.netloc == 'youtu.be':
                return parsed.path[1:]
            else:
                query_params = parse_qs(parsed.query)
                return query_params['v'][0]
        except Exception as e:
            raise InvalidURLError(f"Could not extract video ID from URL: {str(e)}")
    
    async def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Get video information without downloading.
        
        Args:
            url: YouTube URL
            
        Returns:
            Dict containing video metadata
        """
        if not self._is_valid_youtube_url(url):
            raise InvalidURLError(f"Invalid YouTube URL: {url}")
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        try:
            loop = asyncio.get_event_loop()
            
            def _extract_info():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    return ydl.extract_info(url, download=False)
            
            info = await loop.run_in_executor(None, _extract_info)
            
            # Extract relevant metadata
            video_info = {
                'id': info.get('id'),
                'title': info.get('title'),
                'duration': info.get('duration'),
                'description': info.get('description', '')[:500],  # Limit description length
                'uploader': info.get('uploader'),
                'upload_date': info.get('upload_date'),
                'view_count': info.get('view_count'),
                'like_count': info.get('like_count'),
                'thumbnail': info.get('thumbnail'),
                'formats_available': len(info.get('formats', [])),
                'has_audio': any(f.get('acodec') != 'none' for f in info.get('formats', [])),
            }
            
            logger.info(f"Extracted info for video: {video_info['title']} ({video_info['duration']}s)")
            return video_info
            
        except Exception as e:
            logger.error(f"Failed to extract video info from {url}: {str(e)}")
            raise YouTubeDownloadError(f"Failed to get video information: {str(e)}")
    
    async def download_audio(
        self, 
        url: str, 
        quality: str = 'best',
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Download audio from YouTube video.
        
        Args:
            url: YouTube URL
            quality: Audio quality ('best', 'high', 'medium')
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict containing download result information
        """
        if not self._is_valid_youtube_url(url):
            raise InvalidURLError(f"Invalid YouTube URL: {url}")
        
        if quality not in self.audio_formats:
            quality = 'best'
        
        video_id = self._extract_video_id(url)
        session_id = str(uuid.uuid4())
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session_id}_{video_id}_{timestamp}"
        output_path = self.download_dir / f"{filename}.%(ext)s"
        
        # yt-dlp options
        ydl_opts = {
            'outtmpl': str(output_path),
            'quiet': False,
            'no_warnings': False,
            'extractaudio': True,
            'audioformat': 'wav',
            'audioquality': '192',
            **self.audio_formats[quality]
        }
        
        # Add progress hook if callback provided
        if progress_callback:
            def progress_hook(d):
                if d['status'] == 'downloading':
                    # Create a sync wrapper for the async callback
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If we're in an async context, schedule the callback
                            asyncio.create_task(progress_callback({
                                'status': 'downloading',
                                'percent': d.get('_percent_str', '0%'),
                                'speed': d.get('_speed_str', 'N/A'),
                                'eta': d.get('_eta_str', 'N/A')
                            }))
                        else:
                            # If no loop is running, run it synchronously
                            loop.run_until_complete(progress_callback({
                                'status': 'downloading',
                                'percent': d.get('_percent_str', '0%'),
                                'speed': d.get('_speed_str', 'N/A'),
                                'eta': d.get('_eta_str', 'N/A')
                            }))
                    except:
                        # Fallback: ignore async issues for now
                        pass
                elif d['status'] == 'finished':
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(progress_callback({
                                'status': 'finished',
                                'filename': d['filename']
                            }))
                    except:
                        pass
            
            ydl_opts['progress_hooks'] = [progress_hook]
        
        try:
            loop = asyncio.get_event_loop()
            
            def _download():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            
            # Execute download in thread pool
            await loop.run_in_executor(None, _download)
            
            # Find the downloaded file
            expected_file = self.download_dir / f"{filename}.wav"
            if not expected_file.exists():
                # Try to find the file with any extension
                for file_path in self.download_dir.glob(f"{filename}.*"):
                    if file_path.suffix in ['.wav', '.mp3', '.m4a']:
                        expected_file = file_path
                        break
            
            if not expected_file.exists():
                raise YouTubeDownloadError("Downloaded file not found")
            
            # Get file info
            file_size = expected_file.stat().st_size
            
            # Get video info for metadata
            video_info = await self.get_video_info(url)
            
            result = {
                'session_id': session_id,
                'video_id': video_id,
                'filename': expected_file.name,
                'file_path': str(expected_file),
                'file_size': file_size,
                'quality': quality,
                'duration': video_info.get('duration'),
                'title': video_info.get('title'),
                'uploader': video_info.get('uploader'),
                'download_time': datetime.now().isoformat(),
                'url': url
            }
            
            logger.info(f"Successfully downloaded audio: {result['title']} ({file_size / 1024 / 1024:.2f} MB)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to download audio from {url}: {str(e)}")
            # Clean up any partial files
            for file_path in self.download_dir.glob(f"{filename}.*"):
                try:
                    file_path.unlink()
                except:
                    pass
            raise YouTubeDownloadError(f"Download failed: {str(e)}")
    
    async def get_available_formats(self, url: str) -> List[Dict[str, Any]]:
        """
        Get all available formats for a YouTube video.
        
        Args:
            url: YouTube URL
            
        Returns:
            List of available formats with their metadata
        """
        if not self._is_valid_youtube_url(url):
            raise InvalidURLError(f"Invalid YouTube URL: {url}")
        
        try:
            info = await self.get_video_info(url)
            
            # Get detailed format info
            ydl_opts = {'quiet': True, 'no_warnings': True}
            
            loop = asyncio.get_event_loop()
            
            def _get_formats():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    return ydl.extract_info(url, download=False)
            
            detailed_info = await loop.run_in_executor(None, _get_formats)
            
            formats = []
            for fmt in detailed_info.get('formats', []):
                if fmt.get('acodec') != 'none':  # Only audio formats
                    formats.append({
                        'format_id': fmt.get('format_id'),
                        'ext': fmt.get('ext'),
                        'acodec': fmt.get('acodec'),
                        'abr': fmt.get('abr'),  # Audio bitrate
                        'asr': fmt.get('asr'),  # Audio sample rate
                        'filesize': fmt.get('filesize'),
                        'quality': fmt.get('quality'),
                        'format_note': fmt.get('format_note', ''),
                    })
            
            return formats
            
        except Exception as e:
            logger.error(f"Failed to get formats for {url}: {str(e)}")
            raise YouTubeDownloadError(f"Failed to get available formats: {str(e)}")
    
    async def cleanup_old_downloads(self, max_age_hours: int = 24):
        """
        Clean up old downloaded files.
        
        Args:
            max_age_hours: Maximum age of files to keep in hours
        """
        try:
            current_time = datetime.now()
            deleted_count = 0
            
            for file_path in self.download_dir.glob("*"):
                if file_path.is_file():
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_age.total_seconds() > max_age_hours * 3600:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                            logger.info(f"Deleted old download: {file_path.name}")
                        except Exception as e:
                            logger.error(f"Failed to delete {file_path.name}: {str(e)}")
            
            logger.info(f"Cleanup completed: {deleted_count} files deleted")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")


# Factory function for easy instantiation
def create_youtube_downloader(config: Config) -> YouTubeDownloader:
    """Create a YouTube downloader instance."""
    return YouTubeDownloader(config)
