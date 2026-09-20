from __future__ import annotations

import json
import os

from google.cloud import firestore, storage

from dataset_logger import build_eojeol_list

DATASET_BUCKET = os.getenv("DATASET_BUCKET", "malmoi-jeju-dataset-2026")
DATASET_AUDIO_PREFIX = os.getenv("DATASET_AUDIO_PREFIX", "dataset/extracted/Audio")
DATASET_TEXT_PREFIX = os.getenv("DATASET_TEXT_PREFIX", "dataset/extracted/Text")

SAMPLES_COLLECTION = os.getenv("DATASET_FIRESTORE_COLLECTION", "dataset_samples")

STATUSES = ("pending", "approved", "rejected")

_storage_client: storage.Client | None = None
_firestore_client: firestore.Client | None = None


def _client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def samples_collection() -> firestore.CollectionReference:
    global _firestore_client
    if _firestore_client is None:
        _firestore_client = firestore.Client()
    return _firestore_client.collection(SAMPLES_COLLECTION)


def _count(query) -> int:
    return query.count().get()[0][0].value


def get_stats() -> dict:
    col = samples_collection()
    counts = {
        status: _count(col.where(filter=firestore.FieldFilter("status", "==", status)))
        for status in STATUSES
    }
    return {"total": _count(col), **counts}


def list_samples(*, limit: int = 20, offset: int = 0) -> dict:
    """One page of samples, newest first. sample_id embeds a UTC timestamp
    prefix (see dataset_logger.save_training_sample), so id desc == created_at desc."""
    col = samples_collection()
    docs = col.order_by("id", direction=firestore.Query.DESCENDING).offset(offset).limit(limit)
    return {"samples": [d.to_dict() for d in docs.stream()], "total": _count(col)}


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

    blob.metadata = {"status": status}
    blob.upload_from_string(
        json.dumps(record, ensure_ascii=False, indent=2),
        content_type="application/json",
    )
    samples_collection().document(sample_id).set(record)
    return record
