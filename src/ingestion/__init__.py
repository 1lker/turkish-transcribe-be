"""
Ingestion module for data collection and YouTube downloads.
"""

from .youtube_downloader import YouTubeDownloader, create_youtube_downloader

__all__ = [
    'YouTubeDownloader',
    'create_youtube_downloader',
]
