from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone

from google.cloud import storage

logger = logging.getLogger(__name__)

DATASET_BUCKET = os.getenv("DATASET_BUCKET", "malmoi-jeju-dataset-2026")
DATASET_AUDIO_PREFIX = os.getenv("DATASET_AUDIO_PREFIX", "dataset/extracted/Audio")
DATASET_TEXT_PREFIX = os.getenv("DATASET_TEXT_PREFIX", "dataset/extracted/Text")

_storage_client: storage.Client | None = None


def _client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def save_training_sample(
    *,
    audio_path: str,
    jeju_text: str,
    standard_text: str,
    speaker_id: str = "1",
) -> None:
    """Best-effort upload of one (audio, label) pair for future STT training.

    Never raises — dataset collection is supplementary and must not break the
    caller's response. Blocking I/O; run via asyncio.to_thread from async code.
    """
    sample_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    try:
        bucket = _client().bucket(DATASET_BUCKET)

        audio_blob = bucket.blob(f"{DATASET_AUDIO_PREFIX}/{sample_id}.wav")
        audio_blob.upload_from_filename(audio_path)

        record = {
            "id": sample_id,
            "audio_filepath": f"{DATASET_AUDIO_PREFIX}/{sample_id}.wav",
            "form": jeju_text,
            "standard_form": standard_text,
            "dialect_form": jeju_text,
            "speaker_id": speaker_id,
            "note": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        text_blob = bucket.blob(f"{DATASET_TEXT_PREFIX}/{sample_id}.jsonl")
        text_blob.upload_from_string(
            json.dumps(record, ensure_ascii=False) + "\n",
            content_type="application/jsonl",
        )
    except Exception:
        logger.warning("학습 데이터셋 저장 실패 (sample_id=%s)", sample_id, exc_info=True)
