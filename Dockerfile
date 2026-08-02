FROM python:3.11-slim

WORKDIR /app

# librosa가 webm/opus 등 압축 오디오를 디코딩하려면 ffmpeg가 필요하고,
# soundfile 백엔드는 libsndfile1에 의존한다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

# Cloud Run은 컨테이너 외부에서 요청을 보내므로 반드시 0.0.0.0으로 바인딩해야 한다.
CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT}"]
