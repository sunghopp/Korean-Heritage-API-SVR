from __future__ import annotations

import json
import os

from google.cloud import storage

DATASET_BUCKET = os.getenv("DATASET_BUCKET", "malmoi-jeju-dataset-2026")
DATASET_AUDIO_PREFIX = os.getenv("DATASET_AUDIO_PREFIX", "dataset/extracted/Audio")
DATASET_TEXT_PREFIX = os.getenv("DATASET_TEXT_PREFIX", "dataset/extracted/Text")

TIERS = ("auto_approved", "needs_monitoring", "needs_review")

_storage_client: storage.Client | None = None


def _client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def get_stats() -> dict:
    bucket = _client().bucket(DATASET_BUCKET)
    counts = {
        tier: sum(1 for _ in bucket.list_blobs(prefix=f"{DATASET_TEXT_PREFIX}/{tier}/"))
        for tier in TIERS
    }
    return {"total": sum(counts.values()), **counts}


def list_samples(limit: int = 100) -> list[dict]:
    bucket = _client().bucket(DATASET_BUCKET)
    samples: list[dict] = []
    for tier in TIERS:
        for blob in bucket.list_blobs(prefix=f"{DATASET_TEXT_PREFIX}/{tier}/"):
            if blob.name.endswith(".json"):
                samples.append(json.loads(blob.download_as_text()))
    samples.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return samples[:limit]


def get_audio_bytes(tier: str, sample_id: str) -> bytes:
    bucket = _client().bucket(DATASET_BUCKET)
    blob = bucket.blob(f"{DATASET_AUDIO_PREFIX}/{tier}/{sample_id}.wav")
    return blob.download_as_bytes()
