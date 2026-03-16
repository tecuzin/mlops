#!/usr/bin/env bash
set -euo pipefail

SPARK_SERVICE="${SPARK_SERVICE:-spark}"

docker compose exec "${SPARK_SERVICE}" spark-submit /app/lakehouse/jobs/silver_transform.py
docker compose exec "${SPARK_SERVICE}" spark-submit /app/lakehouse/jobs/gold_materialize.py

echo "Compaction refresh finished for Silver and Gold."
