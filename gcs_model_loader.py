from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage

logger = logging.getLogger(__name__)


def resolve_model_path(model_path: str, cache_path: str) -> str:
    """Return a local model directory, downloading a GCS prefix when needed."""
    if not model_path.startswith("gs://"):
        return model_path

    parsed = urlparse(model_path)
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/")
    if not bucket_name or not prefix:
        raise ValueError(f"잘못된 GCS 모델 경로입니다: {model_path}")

    target_dir = Path(cache_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=f"{prefix}/"))
    files = [blob for blob in blobs if not blob.name.endswith("/")]
    if not files:
        raise FileNotFoundError(f"GCS 모델 경로에 파일이 없습니다: {model_path}")

    for blob in files:
        relative_path = Path(blob.name[len(prefix) :].lstrip("/"))
        if not relative_path.parts or any(part == ".." for part in relative_path.parts):
            raise ValueError(f"안전하지 않은 GCS 모델 파일 경로입니다: {blob.name}")
        destination = target_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.stat().st_size != blob.size:
            logger.info("STT 모델 파일 다운로드: gs://%s/%s -> %s", bucket_name, blob.name, destination)
            blob.download_to_filename(str(destination))

    return str(target_dir)


def load_lora_model_path() -> str:
    model_path = os.getenv(
        "LORA_MODEL_PATH",
        "gs://malmoi-jeju-dataset-2026/whisper-model-weights/whisper-jeju-lora-final",
    )
    cache_path = os.getenv("LORA_MODEL_CACHE_PATH", "/tmp/whisper-jeju-lora-final")
    return resolve_model_path(model_path, cache_path)
