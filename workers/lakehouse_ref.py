from __future__ import annotations

import json
from pathlib import Path


def _metadata_filename(namespace: str, table: str) -> str:
    return f"{namespace}_{table}".replace(".", "_") + ".json"


def resolve_lakehouse_dataset_path(
    ref: dict,
    metadata_dir: str | Path,
    role: str = "train",
) -> str:
    if not ref:
        raise ValueError("Missing lakehouse reference")

    namespace = ref.get("namespace")
    table = ref.get("table")
    snapshot_id = ref.get("snapshot_id")
    if not namespace or not table or not snapshot_id:
        raise ValueError("Lakehouse reference must include namespace, table, and snapshot_id")

    metadata_path = Path(metadata_dir) / _metadata_filename(namespace, table)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Lakehouse metadata file not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    expected_table = f"{namespace}.{table}"
    if metadata.get("table") != expected_table:
        raise ValueError(f"Lakehouse metadata table mismatch: expected {expected_table}")
    if metadata.get("snapshot_id") != snapshot_id:
        raise ValueError("Lakehouse snapshot mismatch between run reference and metadata")

    if role == "train":
        path = metadata.get("training_export_path") or metadata.get("path")
    else:
        path = metadata.get("evaluation_export_path") or metadata.get("training_export_path") or metadata.get("path")
    if not path:
        raise ValueError("No readable dataset path in lakehouse metadata")
    return path
