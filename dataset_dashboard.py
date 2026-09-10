from __future__ import annotations

import json
import os

from google.cloud import storage

from dataset_logger import build_eojeol_list

DATASET_BUCKET = os.getenv("DATASET_BUCKET", "malmoi-jeju-dataset-2026")
DATASET_AUDIO_PREFIX = os.getenv("DATASET_AUDIO_PREFIX", "dataset/extracted/Audio")
DATASET_TEXT_PREFIX = os.getenv("DATASET_TEXT_PREFIX", "dataset/extracted/Text")

STATUSES = ("pending", "approved", "rejected")

_storage_client: storage.Client | None = None


def _client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def _fetch_all_records() -> list[dict]:
    bucket = _client().bucket(DATASET_BUCKET)
    records: list[dict] = []
    for blob in bucket.list_blobs(prefix=f"{DATASET_TEXT_PREFIX}/"):
        if blob.name.endswith(".json"):
            records.append(json.loads(blob.download_as_text()))
    records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return records


def get_stats() -> dict:
    records = _fetch_all_records()
    status_counts = {status: 0 for status in STATUSES}
    for record in records:
        status = record.get("status")
        if status in status_counts:
            status_counts[status] += 1
    return {"total": len(records), **status_counts}


def list_samples(limit: int = 100) -> list[dict]:
    return _fetch_all_records()[:limit]


def get_audio_bytes(sample_id: str) -> bytes:
    bucket = _client().bucket(DATASET_BUCKET)
    blob = bucket.blob(f"{DATASET_AUDIO_PREFIX}/{sample_id}.wav")
    return blob.download_as_bytes()


def _resolve_sample_blob(bucket: storage.Bucket, sample_id: str) -> storage.Blob:
    """Locate a sample's text blob under the dataset text prefix.

    Samples are saved flat directly under the prefix, but some pre-existing
    samples still live one directory level deeper (a legacy layout). Try the
    flat path first, then fall back to scanning for a matching filename so
    older samples remain reachable without a data migration.
    """
    flat_blob = bucket.blob(f"{DATASET_TEXT_PREFIX}/{sample_id}.json")
    if flat_blob.exists():
        return flat_blob
    for blob in bucket.list_blobs(prefix=f"{DATASET_TEXT_PREFIX}/"):
        if blob.name.endswith(f"/{sample_id}.json"):
            return blob
    return flat_blob


def update_sample_label(
    *,
    sample_id: str,
    status: str,
    dialect_form: str | None = None,
    standard_form: str | None = None,
) -> dict:
    """Apply a human review decision to one sample's label in place.

    Only called from the dashboard's human review actions, so the reviewer
    is always "human" — system approval happens at save time instead.
    """
    bucket = _client().bucket(DATASET_BUCKET)
    blob = _resolve_sample_blob(bucket, sample_id)
    record = json.loads(blob.download_as_text())

    if dialect_form is not None:
        record["dialect_form"] = dialect_form
        record["form"] = dialect_form
    if standard_form is not None:
        record["standard_form"] = standard_form
    if dialect_form is not None or standard_form is not None:
        record["eojeolList"] = build_eojeol_list(record["dialect_form"], record["standard_form"])

    record["status"] = status
    record["reviewed_by"] = "human"

    blob.upload_from_string(
        json.dumps(record, ensure_ascii=False, indent=2),
        content_type="application/json",
    )
    return record
