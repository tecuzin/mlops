#!/usr/bin/env bash
set -euo pipefail

docker compose exec \
  -e HOME=/tmp \
  -e PYTHONPATH=/app \
  -e USER=spark \
  -e HADOOP_USER_NAME=spark \
  -e SPARK_SUBMIT_OPTS="-Divy.home=/tmp/.ivy2 -Duser.home=/tmp" \
  spark spark-submit /app/lakehouse/jobs/gold_materialize.py
