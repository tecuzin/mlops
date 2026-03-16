from __future__ import annotations

import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from lakehouse.jobs.spark_catalog import build_spark_session, publish_catalog_table
from lakehouse.quality.silver_quality import assert_silver_quality


def _spark() -> SparkSession:
    # build_spark_session wires spark.sql.catalog.nessie and related catalog settings.
    return build_spark_session("lakehouse-silver-transform")


def run() -> None:
    spark = _spark()
    bronze_path = os.getenv("BRONZE_OUTPUT_PATH", "/app/lakehouse/warehouse/bronze/rag_qa")
    silver_path = os.getenv("SILVER_OUTPUT_PATH", "/app/lakehouse/warehouse/silver/rag_qa")

    bronze_df = spark.read.parquet(bronze_path)
    silver_df = (
        bronze_df.select("question", "answer", "context", "source_id", "batch_id", "ingestion_ts")
        .withColumn("question", F.trim(F.col("question")))
        .withColumn("answer", F.trim(F.col("answer")))
        .withColumn("context", F.trim(F.col("context")))
    )
    assert_silver_quality(silver_df)
    publish_catalog_table(silver_df, "silver_rag_qa")
    silver_df.write.mode("overwrite").parquet(silver_path)
    spark.stop()


if __name__ == "__main__":
    run()
