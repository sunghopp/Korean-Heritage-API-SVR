FROM python:3.11-slim

WORKDIR /app

# ffmpeg/libsndfile: uploaded web audio decoding
# git/build-essential: pin and build upstream VITS monotonic_align runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# VITS architecture/runtime. The trained model artifact added by this project is
# only G_*.pth; source is pinned here for reproducible inference.
RUN git clone https://github.com/jaywalnut310/vits.git /opt/vits \
    && cd /opt/vits \
    && git checkout 2e561ba \
    && cd /opt/vits/monotonic_align \
    && mkdir -p monotonic_align \
    && touch monotonic_align/__init__.py \
    && python setup.py build_ext --inplace \
    && rm -rf /opt/vits/.git

COPY . .

ENV PORT=8080 \
    VITS_ROOT=/opt/vits \
    TTS_CONFIG_PATH=/app/tts_config/jeju_vits.json \
    TTS_CHECKPOINT_PATH=gs://malmoi-jeju-dataset-2026/tts/jeju_vits.pth \
    TTS_CHECKPOINT_CACHE_PATH=/tmp/jeju_vits.pth

EXPOSE 8080

# For GPU/CPU-heavy demo inference, Cloud Run concurrency=1 is recommended.
CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT}"]
