"""
Simple test server for YouTube functionality
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="YouTube Test API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class YouTubeInfoResponse(BaseModel):
    success: bool
    video_id: str
    title: str
    duration: Optional[int] = None
    description: str = ""
    uploader: Optional[str] = None
    upload_date: Optional[str] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    thumbnail: Optional[str] = None
    has_audio: bool = True
    formats_count: int = 0

class YouTubeFormatInfo(BaseModel):
    format_id: str
    ext: str
    acodec: Optional[str] = None
    abr: Optional[float] = None
    asr: Optional[int] = None
    filesize: Optional[int] = None
    quality: Optional[str] = None
    format_note: str = ""

class YouTubeFormatsResponse(BaseModel):
    success: bool
    formats: List[YouTubeFormatInfo]
    total_formats: int

@app.get("/")
async def root():
    return {"message": "YouTube Test API is running"}

@app.get("/youtube/info", response_model=YouTubeInfoResponse)
async def get_youtube_info(url: str):
    """Mock YouTube info endpoint"""
    if "youtube.com" not in url and "youtu.be" not in url:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    return YouTubeInfoResponse(
        success=True,
        video_id="test123",
        title="Test Video Title",
        duration=300,
        description="This is a test video description",
        uploader="Test Channel",
        upload_date="2025-01-01",
        view_count=1000000,
        like_count=50000,
        thumbnail="https://img.youtube.com/vi/test123/maxresdefault.jpg",
        has_audio=True,
        formats_count=5
    )

@app.get("/youtube/formats", response_model=YouTubeFormatsResponse)
async def get_youtube_formats(url: str):
    """Mock YouTube formats endpoint"""
    if "youtube.com" not in url and "youtu.be" not in url:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    formats = [
        YouTubeFormatInfo(
            format_id="140",
            ext="m4a",
            acodec="mp4a.40.2",
            abr=128.0,
            asr=44100,
            filesize=5242880,
            quality="medium",
            format_note="128k"
        ),
        YouTubeFormatInfo(
            format_id="251",
            ext="webm",
            acodec="opus",
            abr=160.0,
            asr=48000,
            filesize=6291456,
            quality="high",
            format_note="160k"
        )
    ]
    
    return YouTubeFormatsResponse(
        success=True,
        formats=formats,
        total_formats=len(formats)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
