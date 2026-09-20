"""One-off: load every existing GCS Text/**.json record into Firestore.

Run from backend/: python scripts/backfill_firestore.py
Idempotent (doc id = sample_id, set() overwrites).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_dashboard import DATASET_BUCKET, DATASET_TEXT_PREFIX, _client, samples_collection

bucket = _client().bucket(DATASET_BUCKET)
col = samples_collection()
n = 0
for blob in bucket.list_blobs(prefix=f"{DATASET_TEXT_PREFIX}/"):
    if not blob.name.endswith(".json"):
        continue
    record = json.loads(blob.download_as_text())
    sample_id = blob.name.rsplit("/", 1)[-1].removesuffix(".json")
    record.setdefault("id", sample_id)
    record.setdefault("status", (blob.metadata or {}).get("status", "pending"))
    col.document(sample_id).set(record)
    n += 1
print(f"backfilled {n} samples")
