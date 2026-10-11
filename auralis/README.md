# Auralis — Real-Time Voice-to-Voice Emotion Engine

## Overview

Auralis is a real-time conversational AI pipeline built around a Crisis Negotiation
Training Simulator use case: a person speaks into their microphone, the system
transcribes what they said, detects their emotional tone from the audio itself,
generates an emotion-aware response with a local LLM, converts that response to
speech, and streams it back — while supporting interruption if the user starts
talking while the AI is still speaking.

Pipeline: **Browser mic → WebRTC (aiortc) → Silero VAD → Faster-Whisper → Wav2Vec2
emotion → Llama 3 (Ollama) → XTTSv2/Bark → WebRTC → Browser speaker.**

## Features

- Persistent WebRTC audio connection (not repeated HTTP uploads)
- Silero VAD-based speech start/stop detection
- Faster-Whisper speech-to-text
- Wav2Vec2-based emotion classification (RAVDESS label set: angry, calm, disgust,
  fearful, happy, neutral, sad, surprised)
- Llama 3 via Ollama, prompted with transcript + emotion + conversation history
- XTTSv2 (or Bark) text-to-speech, chunked for streaming playback
- Interruption handling — user speech cancels in-flight AI response/TTS
- Live waveform visualizer for both user and AI audio
- Real, measured latency breakdown (STT / TTFT / LLM / TTS / first-audio / total)
  — nothing here is a hard-coded or claimed number

## Architecture

```
React (WebRTC + waveform + UI)
        ↕ WebRTC (audio) + DataChannel (control JSON)
FastAPI + aiortc backend
        ├─ Silero VAD        (speech segmentation)
        ├─ Faster-Whisper    (STT)
        ├─ Wav2Vec2          (emotion)
        ├─ Ollama / Llama 3  (LLM, HTTP, separate service)
        └─ XTTSv2 / Bark     (TTS)
```

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, native WebRTC APIs, Web Audio API (waveform) |
| Backend | Python, FastAPI, aiortc, asyncio |
| STT | Faster-Whisper |
| VAD | Silero VAD (via `torch.hub`) |
| Emotion | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (Transformers) |
| LLM | Llama 3 via Ollama (local, HTTP) |
| TTS | Coqui TTS (XTTSv2) or Bark |

---

## Installation

### Prerequisites

```bash
python --version     # 3.10 or 3.11 recommended
node --version        # v18+ / v20+
npm --version
ollama --version      # install from https://ollama.com if missing
git --version
```

### 1. Clone and set up Python environment

```bash
git clone <your-repo-url> auralis
cd auralis
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r backend/requirements.txt
```

> **PyAV / aiortc note:** `av` (PyAV) needs FFmpeg's shared libraries on your system.
> On Ubuntu/Debian: `sudo apt install ffmpeg libavdevice-dev libavfilter-dev libopus-dev libvpx-dev libsrtp2-dev`.
> On macOS: `brew install ffmpeg`. On Windows, PyAV's wheels usually bundle FFmpeg —
> if install fails, install FFmpeg separately and ensure it's on PATH.

### 2. Ollama setup

```bash
ollama pull llama3
ollama serve          # usually already running as a background service
curl http://localhost:11434/api/tags   # sanity check it responds
```

### 3. TTS speaker reference (required for XTTSv2)

XTTSv2 is a zero-shot voice-cloning model — it has no built-in default voice, it needs
a short reference clip to clone from:

```bash
mkdir -p models
# Record or supply a clean 5-10 second speech .wav (any voice) here:
# models/speaker_reference.wav
```

### 4. Frontend setup

```bash
cd frontend
npm install
cp ../.env.example .env   # then edit VITE_BACKEND_URL if needed
```

### 5. Environment variables

```bash
cd ..   # project root
cp .env.example .env
# edit .env: at minimum confirm OLLAMA_URL, WHISPER_MODEL, TTS_ENGINE
```

**Never commit `.env`.** Only `.env.example` (with placeholders) is checked in.

---

## Running locally

**Terminal 1 — Ollama** (if not already running):
```bash
ollama serve
```

**Terminal 2 — Backend** (run from project root, not from `backend/`):
```bash
source venv/bin/activate
uvicorn backend.main:app --reload --port 8000
```
First startup downloads Whisper/Silero/emotion/TTS model weights — this can take
several minutes and needs internet access. Subsequent startups use the local cache.

**Terminal 3 — Frontend**:
```bash
cd frontend
npm run dev
```

Open **http://localhost:5173**, click **Start Conversation**, allow microphone access,
and speak.

---

## Testing

```bash
pip install pytest pytest-asyncio soundfile
pytest tests/ -v
```

- `test_stt.py`, `test_vad.py`, `test_emotion.py` — model-level unit tests; some
  cases are skipped unless you add real audio fixtures under `tests/fixtures/`
  (instructions are in each test file's docstring).
- `test_llm.py` — requires Ollama running with the configured model pulled.
- **WebRTC / interruption / TTS** are best verified manually end-to-end via the
  browser UI (automated WebRTC testing needs a headless-browser harness, which is
  a reasonable future addition, not included here).

---

## Deployment

### Reality check first (read this before picking a host)

This is **not** a normal React website you can drop on generic static/serverless
hosting. It needs:

- A **persistent Python process** (not a serverless function) for aiortc/WebRTC
- **WebSocket/WebRTC support**, including UDP for actual media (many free PaaS tiers
  only proxy HTTP/TCP, which breaks WebRTC media — check specifically for this)
- Enough **RAM** to hold Whisper + Silero VAD + the emotion model + XTTS simultaneously
  (realistically 6–8GB+ even with small model variants, on CPU)
- **GPU strongly recommended** for XTTS and Whisper to run at usable speed — CPU-only
  will work but each conversational turn will likely take several seconds to tens of
  seconds, not the sub-800ms the source project document targets (that target assumes
  GPU-accelerated streaming inference, which this reference implementation does not
  yet fully achieve even on GPU — see Limitations)

**Practical options:** a GPU-enabled VM (e.g. a cloud provider's GPU instance) or a
dedicated/rented GPU server running the Docker Compose stack below. Do **not** expect
a free-tier serverless host to run this reliably — if you try one and it can't keep
the models loaded/warm or doesn't support raw UDP for WebRTC, that's why, and the fix
is a persistent GPU-capable host, not swapping XTTS/Whisper for an unrelated cloud API.

### Deployment architecture

```
Internet → React (static hosting) → HTTPS/WSS → FastAPI+aiortc (persistent GPU/CPU server)
                                                        ├─ Whisper, Silero VAD, Wav2Vec2
                                                        ├─ Ollama (same or adjacent server)
                                                        └─ XTTS/Bark
```

### Backend + Ollama (Docker Compose)

```bash
docker compose up --build -d
docker compose logs -f backend     # watch model loading progress
curl http://localhost:8000/health
```

Set production env vars (`ALLOWED_ORIGINS`, `OLLAMA_URL`, etc.) via a `.env` file next
to `docker-compose.yml`, or your platform's secret/env manager — never hard-coded.

### Frontend

```bash
cd frontend
npm run build
```
Deploy the `dist/` folder to your static host of choice, with:
```
VITE_BACKEND_URL=https://your-backend-domain.com
```
set at build time. Never hard-code `localhost` URLs into a production build.

### HTTPS / WSS

Browsers require a secure context for microphone access in production (`https://`,
not `http://`). Put the backend behind a reverse proxy (nginx/Caddy/Traefik) that
terminates TLS and forwards WebSocket/WebRTC signaling traffic; WebRTC's actual media
still flows via UDP directly between browser and server (ensure your host's firewall
allows the UDP port range aiortc negotiates, or configure a TURN server if the host
is behind restrictive NAT).

### CORS

`ALLOWED_ORIGINS` in `.env` — set this to your **actual production frontend domain**
in production. The default `.env.example` only allows `localhost:5173` for
development; do not deploy with `allow_origins=["*"]`.

---

## Environment Variables

See `.env.example` for the full list with defaults (network URLs, Ollama/LLM config,
Whisper model/device, emotion model, TTS engine/device/speaker reference, VAD
thresholds, and the frontend's `VITE_BACKEND_URL`).

---

## Project Structure

```
auralis/
├── backend/
│   ├── main.py, config.py, requirements.txt, Dockerfile
│   ├── api/routes.py                 # WebRTC offer/answer + /health
│   ├── services/                     # vad, stt, emotion, llm, tts, audio, streaming
│   ├── models/model_manager.py       # loads every model once at startup
│   └── utils/                        # audio conversion, latency tracking
├── frontend/
│   ├── src/App.jsx, main.jsx, styles.css
│   ├── src/components/               # Controls, Conversation, Emotion, Status, Waveform, Latency
│   └── src/services/                 # webrtc.js, api.js
├── tests/                            # test_stt, test_vad, test_emotion, test_llm
├── models/                           # speaker_reference.wav goes here (gitignored)
├── .env.example, .gitignore, docker-compose.yml
```

---

## Performance Metrics

The app measures and displays (only when actually recorded — never estimated):
STT latency, Time-to-First-Token (TTFT), LLM latency, TTS latency, first-audio
latency, and total turn latency. Expect these to vary heavily by hardware; report
your own measured numbers rather than assuming the document's sub-800ms target
is met without confirming it on your own setup.

## Limitations

- **CPU-only latency is not real-time-conversational.** Whisper `base` and
  especially XTTSv2 are slow on CPU. Sub-second total latency realistically needs
  GPU inference for every stage.
- **TTS is not incrementally streamed at the model level.** XTTS/Bark generate the
  full utterance, which is then chunked for delivery — first-audio latency is
  improved slightly, but total TTS compute time is not reduced. True incremental
  synthesis needs XTTS's lower-level `inference_stream` API.
- **Emotion labels are fixed by the pretrained model** (RAVDESS's 8 classes) — no
  custom classes are invented.
- **Interruption handling** cancels the current LLM/TTS task and clears queued audio,
  but does not yet fade out audio already buffered in the browser's playback pipeline
  — there may be a brief tail of audio after an interrupt.
- **This code has not been executed end-to-end** in the environment that produced it
  (no GPU, no access to Hugging Face/Ollama model downloads there). Expect to do some
  environment-specific debugging, particularly around PyAV/FFmpeg system dependencies
  and the XTTS speaker-reference requirement.

## Future Improvements

- Lower-level XTTS streaming inference for true incremental TTS
- TURN server config for browsers behind restrictive NAT
- GPU batching across concurrent sessions
- Automated WebRTC integration tests (headless browser harness)
- Audio fade-out on interruption instead of a hard stop

---

## Security & Housekeeping

- All secrets/URLs come from `.env` (never committed) — see `.env.example`.
- CORS is restricted to `ALLOWED_ORIGINS`, not `*`, in production.
- No microphone audio is written to disk — everything is processed in-memory and
  discarded per turn.
- `.gitignore` excludes `venv/`, `node_modules/`, `__pycache__/`, `.env`, model
  weight files, and any `.wav`/`.webm` audio.

## Production Checklist

```
[ ] Frontend builds (npm run build)
[ ] Backend starts, /health returns {"status": "ok"}
[ ] Microphone permission works in browser
[ ] WebRTC connects (connectionState -> "connected")
[ ] VAD triggers speech_start/speech_end events
[ ] Whisper produces a transcript
[ ] Emotion label + confidence appear
[ ] Ollama reachable, Llama 3 responds
[ ] TTS produces audio, playback works
[ ] Interrupting AI mid-speech stops it
[ ] Latency numbers appear and look real (not zero/undefined)
[ ] HTTPS in production
[ ] WSS/WebRTC media works through your host's firewall
[ ] CORS restricted to your real frontend domain
[ ] No secrets committed to git
[ ] No persistent audio storage
```
