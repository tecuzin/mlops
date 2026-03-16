from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from pyspark.sql import SparkSession

from lakehouse.jobs.spark_catalog import (
    build_spark_session,
    catalog_name,
    catalog_namespace,
    catalog_ref,
    publish_catalog_table,
)


def _spark() -> SparkSession:
    # build_spark_session wires spark.sql.catalog.nessie and related catalog settings.
    return build_spark_session("lakehouse-gold-materialize")


def run() -> None:
    spark = _spark()
    silver_path = os.getenv("SILVER_OUTPUT_PATH", "/app/lakehouse/warehouse/silver/rag_qa")
    gold_path = os.getenv("GOLD_OUTPUT_PATH", "/app/lakehouse/warehouse/gold/rag_qa_train_ready")
    metadata_dir = Path(os.getenv("GOLD_METADATA_DIR", "/app/lakehouse/metadata"))
    table_name = os.getenv("GOLD_TABLE_NAME", "rag_qa_train_ready")
    namespace = os.getenv("GOLD_TABLE_NAMESPACE", catalog_namespace())
    reference = os.getenv("GOLD_TABLE_REFERENCE", catalog_ref())
    catalog = os.getenv("GOLD_TABLE_CATALOG", catalog_name())

    silver_df = spark.read.parquet(silver_path)
    gold_df = silver_df.select("question", "answer", "context")
    publish_catalog_table(gold_df, table_name)
    gold_df.write.mode("overwrite").parquet(gold_path)

    snapshot_id = os.getenv("GOLD_SNAPSHOT_ID", "").strip() or dt.datetime.utcnow().strftime("snapshot-%Y%m%d%H%M%S")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "catalog": catalog,
        "table": f"{namespace}.{table_name}",
        "reference": reference,
        "snapshot_id": snapshot_id,
        "catalog_commit_id": os.getenv("NESSIE_COMMIT_ID", snapshot_id),
        "materialized_at": dt.datetime.utcnow().isoformat() + "Z",
        "path": gold_path,
        "training_export_path": "/app/data/train/rag_qa_train.jsonl",
        "evaluation_export_path": "/app/data/eval/ragas_eval.jsonl",
    }
    (metadata_dir / f"{namespace}_{table_name}.json").write_text(json.dumps(metadata, indent=2))
    spark.stop()


if __name__ == "__main__":
    run()
