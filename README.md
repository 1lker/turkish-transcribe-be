# Turkish Education Transcription System

🎓 **AI-Powered Turkish Education Transcription System** - Powered by OpenAI Whisper

## 🌟 Features

- **🇹🇷 Turkish Language Optimized**: Fine-tuned for Turkish educational content
- **🎥 YouTube Integration**: Direct download and transcription from YouTube videos  
- **⚡ Real-time Progress**: Live WebSocket updates during transcription
- **🎯 Multiple Output Formats**: JSON, SRT (subtitles), and plain text
- **🔧 Advanced Audio Processing**: VAD (Voice Activity Detection) and normalization
- **🌐 Modern Web Interface**: React/Next.js frontend with real-time updates
- **📊 Multiple Model Support**: Tiny, Base, Small, Medium, Large Whisper models
- **⚙️ Flexible API**: RESTful API with comprehensive endpoints

## 🏗️ Architecture

### Backend (FastAPI)
- **Audio Processing**: VAD, normalization, format conversion
- **Transcription Engine**: OpenAI Whisper with Turkish optimization  
- **WebSocket Support**: Real-time progress updates
- **YouTube Integration**: yt-dlp for video/audio download
- **File Management**: Upload, processing, and result storage

### Frontend (Next.js 14)
- **Modern UI**: Tailwind CSS with shadcn/ui components
- **Real-time Updates**: WebSocket integration for live progress
- **File Management**: Drag & drop upload with progress tracking
- **Result Viewer**: Formatted transcription display with download options

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- ffmpeg

### Backend Setup
```bash
# Clone and setup
git clone <repo-url>
cd turkish-edu-transcription

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 📡 API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `POST /upload` - File upload
- `POST /transcribe/{file_id}` - Start transcription
- `GET /task/{task_id}` - Get transcription status
- `WebSocket /ws/{task_id}` - Real-time updates

### YouTube Integration
- `GET /youtube/info?url=<url>` - Get video info
- `POST /youtube/download` - Download video/audio
- `WebSocket /youtube/download/{session_id}/ws` - Download progress

## 🔧 Configuration

Configuration is managed through `src/core/config.py`:

```python
# Model settings
WHISPER_MODEL_SIZE = "base"  # tiny, base, small, medium, large
DEVICE = "auto"  # auto, cpu, cuda

# Processing settings
BATCH_SIZE = 4
APPLY_VAD = True
NORMALIZE_AUDIO = True

# Server settings
HOST = "0.0.0.0"
PORT = 8000
```

## 📁 Project Structure

```
turkish-edu-transcription/
├── src/
│   ├── api/           # FastAPI application
│   ├── core/          # Configuration and utilities
│   ├── transcription/ # Whisper engine and pipeline
│   ├── ingestion/     # Data ingestion
│   └── processing/    # Audio processing
├── frontend/          # Next.js web interface
├── data/              # Data storage
│   ├── raw/          # Original audio files
│   ├── processed/    # Processed audio files
│   └── transcripts/  # Output transcriptions
├── models/           # Whisper model storage
├── logs/             # Application logs
└── configs/          # Configuration files
```

## 🎯 Usage Examples

### Python API Client
```python
import requests

# Upload file
files = {'file': open('audio.wav', 'rb')}
response = requests.post('http://localhost:8000/upload', files=files)
file_id = response.json()['file_id']

# Start transcription
transcription_request = {
    "model_size": "base",
    "language": "tr",
    "apply_vad": True,
    "normalize_audio": True
}
response = requests.post(
    f'http://localhost:8000/transcribe/{file_id}', 
    json=transcription_request
)
task_id = response.json()['task_id']

# Check status
response = requests.get(f'http://localhost:8000/task/{task_id}')
print(response.json())
```

### WebSocket Real-time Updates
```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/${taskId}`);

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Progress:', data);
};
```

## 🔍 Features Detail

### Audio Processing Pipeline
1. **Format Validation**: Supports WAV, MP3, MP4, WEBM, M4A
2. **VAD Processing**: Removes silence for faster processing
3. **Audio Normalization**: Optimizes volume levels
4. **Whisper Transcription**: Multi-model support with language detection

### Real-time Progress Tracking
- WebSocket-based live updates
- Stage-by-stage progress reporting  
- Error handling and retry mechanisms
- Connection management for multiple clients

### YouTube Integration
- Direct URL processing
- Format selection (audio quality)
- Background download with progress
- Automatic transcription trigger

## 🧪 Testing

```bash
# Run tests
pytest

# Test specific components
pytest src/tests/test_transcription.py
pytest src/tests/test_api.py
```

## 📊 Performance

- **Real-time Factor**: ~0.3x (3 minutes audio → 1 minute processing)
- **Supported File Size**: Up to 2GB
- **Concurrent Processing**: Multiple transcription tasks
- **Memory Usage**: ~2GB RAM for base model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper**: State-of-the-art speech recognition
- **yt-dlp**: YouTube video/audio downloader
- **FastAPI**: Modern Python web framework
- **Next.js**: React framework for production
- **Tailwind CSS**: Utility-first CSS framework

---

Made with ❤️ for Turkish Education Community
