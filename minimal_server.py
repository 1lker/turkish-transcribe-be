"""
Minimal backend for YouTube functionality only
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import os

app = FastAPI(title="YouTube Download API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "YouTube Download API is running"}

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a file from the raw data directory."""
    try:
        # Check current directory structure
        data_dir = Path("./data/raw")
        file_path = data_dir / filename
        
        print(f"Looking for file: {file_path}")
        print(f"File exists: {file_path.exists()}")
        
        if not file_path.exists():
            # List available files for debugging
            if data_dir.exists():
                files = list(data_dir.glob("*"))
                print(f"Available files: {[f.name for f in files]}")
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='audio/wav'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
