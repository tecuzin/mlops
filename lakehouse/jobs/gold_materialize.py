from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from pyspark.sql import SparkSession


def _spark() -> SparkSession:
    return (
        SparkSession.builder.appName("lakehouse-gold-materialize")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def run() -> None:
    spark = _spark()
    silver_path = os.getenv("SILVER_OUTPUT_PATH", "/app/lakehouse/warehouse/silver/rag_qa")
    gold_path = os.getenv("GOLD_OUTPUT_PATH", "/app/lakehouse/warehouse/gold/rag_qa_train_ready")
    metadata_dir = Path(os.getenv("GOLD_METADATA_DIR", "/app/lakehouse/metadata"))

    silver_df = spark.read.parquet(silver_path)
    gold_df = silver_df.select("question", "answer", "context")
    gold_df.write.mode("overwrite").parquet(gold_path)

    snapshot_id = dt.datetime.utcnow().strftime("snapshot-%Y%m%d%H%M%S")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "table": "gold.rag_qa_train_ready",
        "reference": "main",
        "snapshot_id": snapshot_id,
        "materialized_at": dt.datetime.utcnow().isoformat() + "Z",
        "path": gold_path,
        "training_export_path": "/app/data/train/rag_qa_train.jsonl",
        "evaluation_export_path": "/app/data/eval/ragas_eval.jsonl",
    }
    (metadata_dir / "gold_rag_qa_train_ready.json").write_text(json.dumps(metadata, indent=2))
    spark.stop()


if __name__ == "__main__":
    run()
