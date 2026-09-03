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


def build_eojeol_list(dialect_text: str, standard_text: str) -> list[dict]:
    """Split into whitespace-delimited eojeol and pair by position.

    We don't get true word-level alignment from Gemini (only whole-sentence
    dialect/standard text), so this positional pairing is a heuristic and can
    misalign when the two sentences don't have the same word count.
    """
    dialect_words = dialect_text.split()
    standard_words = standard_text.split()

    eojeol_list = []
    for idx, word in enumerate(dialect_words):
        standard_word = standard_words[idx] if idx < len(standard_words) else None
        eojeol_list.append(
            {
                "id": idx + 1,
                "eojeol": word,
                "standard": standard_word,
                "isDialect": (word != standard_word) if standard_word is not None else None,
            }
        )
    return eojeol_list


def _compute_tier(confidence: float) -> str:
    """Classify STT confidence into a fixed tier. Never changes after creation
    — unlike review_status, which tracks human review workflow separately.
    """
    if confidence >= 0.8:
        return "tier1"
    if confidence >= 0.6:
        return "tier2"
    return "tier3"


def _initial_review_status(tier: str) -> str:
    """tier1 is high-confidence enough to skip review; tier2/tier3 start
    out awaiting a human review decision."""
    return "not_required" if tier == "tier1" else "unreviewed"


def save_training_sample(
    *,
    audio_path: str,
    jeju_text: str,
    standard_text: str,
    confidence: float,
    speaker_id: str = "1",
) -> None:
    """Best-effort upload of one (audio, label) pair for future STT training.

    Never raises — dataset collection is supplementary and must not break the
    caller's response. Blocking I/O; run via asyncio.to_thread from async code.
    """
    sample_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    tier = _compute_tier(confidence)
    review_status = _initial_review_status(tier)
    try:
        bucket = _client().bucket(DATASET_BUCKET)

        audio_blob_path = f"{DATASET_AUDIO_PREFIX}/{sample_id}.wav"
        audio_blob = bucket.blob(audio_blob_path)
        audio_blob.upload_from_filename(audio_path)

        record = {
            "id": sample_id,
            "audio_filepath": audio_blob_path,
            "form": jeju_text,
            "standard_form": standard_text,
            "dialect_form": jeju_text,
            "speaker_id": speaker_id,
            "note": "",
            "eojeolList": build_eojeol_list(jeju_text, standard_text),
            "confidence": round(confidence, 4),
            "tier": tier,
            "review_status": review_status,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        text_blob = bucket.blob(f"{DATASET_TEXT_PREFIX}/{tier}/{sample_id}.json")
        text_blob.upload_from_string(
            json.dumps(record, ensure_ascii=False, indent=2),
            content_type="application/json",
        )
    except Exception:
        logger.warning("학습 데이터셋 저장 실패 (sample_id=%s)", sample_id, exc_info=True)
