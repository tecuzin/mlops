from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def assert_silver_quality(df: DataFrame) -> None:
    required_columns = ("question", "answer", "context", "source_id", "batch_id", "ingestion_ts")
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Silver schema missing columns: {missing}")

    checks = {
        "question": df.filter(F.col("question").isNull() | (F.length(F.trim(F.col("question"))) == 0)).count(),
        "answer": df.filter(F.col("answer").isNull() | (F.length(F.trim(F.col("answer"))) == 0)).count(),
        "context": df.filter(F.col("context").isNull() | (F.length(F.trim(F.col("context"))) == 0)).count(),
    }
    invalid = {k: v for k, v in checks.items() if v > 0}
    if invalid:
        raise ValueError(f"Silver quality checks failed: {invalid}")
