from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def _spark() -> SparkSession:
    return (
        SparkSession.builder.appName("lakehouse-bronze-ingest")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def run() -> None:
    spark = _spark()

    source_path = os.getenv("BRONZE_SOURCE_PATH", "/app/data/train/rag_qa_train.jsonl")
    output_path = os.getenv("BRONZE_OUTPUT_PATH", "/app/lakehouse/warehouse/bronze/rag_qa")
    source_id = os.getenv("BRONZE_SOURCE_ID", "rag_qa_train")
    batch_id = os.getenv("BRONZE_BATCH_ID", dt.datetime.utcnow().strftime("%Y%m%d%H%M%S"))

    if not Path(source_path).exists():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")

    df = spark.read.json(source_path)
    bronze_df = (
        df.withColumn("ingestion_ts", F.current_timestamp())
        .withColumn("source_id", F.lit(source_id))
        .withColumn("batch_id", F.lit(batch_id))
    )

    bronze_df.write.mode("append").parquet(output_path)
    spark.stop()


if __name__ == "__main__":
    run()
