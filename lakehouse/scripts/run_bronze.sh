#!/usr/bin/env bash
set -euo pipefail

docker compose exec \
  -e HOME=/tmp \
  -e PYTHONPATH=/app \
  -e USER=spark \
  -e HADOOP_USER_NAME=spark \
  -e SPARK_SUBMIT_OPTS="-Divy.home=/tmp/.ivy2 -Duser.home=/tmp" \
  spark spark-submit \
  --packages "${SPARK_CATALOG_PACKAGES:-org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1,org.projectnessie.nessie-integrations:nessie-spark-extensions-3.5_2.12:0.104.5,software.amazon.awssdk:bundle:2.25.50,software.amazon.awssdk:url-connection-client:2.25.50}" \
  /app/lakehouse/jobs/bronze_ingest.py
