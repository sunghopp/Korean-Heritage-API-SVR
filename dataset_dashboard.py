from __future__ import annotations

import json
import os

from google.cloud import storage

from dataset_logger import build_eojeol_list

DATASET_BUCKET = os.getenv("DATASET_BUCKET", "malmoi-jeju-dataset-2026")
DATASET_AUDIO_PREFIX = os.getenv("DATASET_AUDIO_PREFIX", "dataset/extracted/Audio")
DATASET_TEXT_PREFIX = os.getenv("DATASET_TEXT_PREFIX", "dataset/extracted/Text")

TIERS = ("tier1", "tier2", "tier3")

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


def get_audio_bytes(sample_id: str) -> bytes:
    bucket = _client().bucket(DATASET_BUCKET)
    blob = bucket.blob(f"{DATASET_AUDIO_PREFIX}/{sample_id}.wav")
    return blob.download_as_bytes()


def update_sample_label(
    *,
    tier: str,
    sample_id: str,
    review_status: str,
    dialect_form: str | None = None,
    standard_form: str | None = None,
) -> dict:
    """Apply a human review decision to one sample's label in place.

    Tier never changes (it's a fixed classification from creation time), so
    this updates the file at its existing path — no move/delete involved.
    """
    bucket = _client().bucket(DATASET_BUCKET)
    blob = bucket.blob(f"{DATASET_TEXT_PREFIX}/{tier}/{sample_id}.json")
    record = json.loads(blob.download_as_text())

    if dialect_form is not None:
        record["dialect_form"] = dialect_form
        record["form"] = dialect_form
    if standard_form is not None:
        record["standard_form"] = standard_form
    if dialect_form is not None or standard_form is not None:
        record["eojeolList"] = build_eojeol_list(record["dialect_form"], record["standard_form"])

    record["review_status"] = review_status

    blob.upload_from_string(
        json.dumps(record, ensure_ascii=False, indent=2),
        content_type="application/json",
    )
    return record
