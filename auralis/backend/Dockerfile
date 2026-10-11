FROM python:3.11-slim

# System deps: ffmpeg/libav for aiortc+PyAV audio decoding, build tools for
# torch/TTS native extensions, libsndfile for audio I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavdevice-dev \
    libavfilter-dev \
    libopus-dev \
    libvpx-dev \
    libsrtp2-dev \
    libsndfile1 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY models ./models

# Models are NOT baked into the image (see docker-compose.yml comment) -
# they download to a mounted volume on first startup instead.

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
