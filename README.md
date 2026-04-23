# Turkish Transcribe — Backend

> Production speech-to-text service for Turkish educational content. Whisper-powered, FastAPI-served, real-time.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Whisper](https://img.shields.io/badge/Whisper-OpenAI-412991?style=flat-square)](https://github.com/openai/whisper)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](Dockerfile.backend)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

---

## What this does

A self-hosted Turkish speech-to-text service tuned for **educational content** — lectures, classroom recordings, YouTube tutorials. Built around OpenAI Whisper with audio-quality preprocessing, real-time WebSocket progress, and multi-format output (JSON / SRT / TXT).

Frontend lives at [turkish-transcribe-fe](https://github.com/1lker/turkish-transcribe-fe).

---

## Why it exists

General-purpose Turkish transcription tools choke on classroom audio: bad mics, code-switching with English technical terms, long monologues, slide-flip noise. This service ships the preprocessing + model selection that makes those usable.

---

## Features

| Capability | Detail |
|---|---|
| **Turkish-optimized** | Whisper model selection + decoding params tuned for Turkish-language audio |
| **YouTube ingest** | Direct URL → audio extract via `yt-dlp` |
| **Real-time progress** | WebSocket stream of chunk-level transcription state |
| **Multi-format output** | JSON, SRT subtitles, plain text |
| **Audio preprocessing** | Voice Activity Detection (VAD) + loudness normalization |
| **Model selection** | Tiny / Base / Small / Medium / Large — speed vs accuracy trade |
| **Containerized** | Dockerfile + nginx reverse proxy + docker-compose |

---

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        A[YouTube URL] --> B[yt-dlp extract]
        C[File upload] --> D[Validation]
        B --> E[Audio normalization]
        D --> E
    end

    subgraph Processing
        E --> F[VAD chunking]
        F --> G[Whisper model]
        G --> H[Post-processing]
    end

    subgraph Output
        H --> I[JSON]
        H --> J[SRT]
        H --> K[TXT]
    end

    G -.->|progress| L[WebSocket]

    style A fill:#1a1a1f,stroke:#f97316,color:#fff
    style C fill:#1a1a1f,stroke:#f97316,color:#fff
    style I fill:#1a1a1f,stroke:#34d399,color:#fff
    style J fill:#1a1a1f,stroke:#34d399,color:#fff
    style K fill:#1a1a1f,stroke:#34d399,color:#fff
```

---

## Project structure

```
src/
├── api/             # FastAPI routes + WebSocket handlers
├── core/            # Config, logging, model loading
├── ingestion/       # YouTube + file upload pipelines
├── processing/      # VAD, normalization, chunking
├── storage/         # Result persistence
└── transcription/   # Whisper wrapper + format exporters
```

---

## Quick start

### Local

```bash
git clone https://github.com/1lker/turkish-transcribe-be.git
cd turkish-transcribe-be
pip install -r requirements.txt

# Run server
python minimal_server.py
# → http://localhost:8000
```

### Docker

```bash
docker compose up -d
```

### CLI

```bash
python cli.py --input lecture.mp4 --model medium --format srt
```

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/transcribe/file` | `POST` | Upload audio/video for transcription |
| `/transcribe/youtube` | `POST` | Transcribe from YouTube URL |
| `/ws/progress/{job_id}` | `WS` | Real-time progress stream |
| `/results/{job_id}` | `GET` | Fetch finished transcript |
| `/health` | `GET` | Service health check |

---

## Tech stack

- **Server:** FastAPI + Uvicorn + WebSockets
- **Reverse proxy:** Nginx
- **ASR:** OpenAI Whisper (`tiny` → `large`)
- **Audio:** ffmpeg, librosa, webrtcvad
- **Ingest:** yt-dlp
- **Container:** Docker + docker-compose

---

## Author

**İlker Yörü** — CTO @ [Mindra](https://mindra.co)
[GitHub](https://github.com/1lker) · [LinkedIn](https://linkedin.com/in/ilker-yoru) · [ilkeryoru.com](https://ilkeryoru.com)

## License

MIT
