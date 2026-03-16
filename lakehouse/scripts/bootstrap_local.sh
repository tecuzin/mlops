#!/usr/bin/env bash
set -euo pipefail

MINIO_ALIAS="${MINIO_ALIAS:-local}"
MINIO_ENDPOINT="${LAKEHOUSE_S3_ENDPOINT:-http://localhost:9000}"
MINIO_USER="${MINIO_ROOT_USER:-minio}"
MINIO_PASSWORD="${MINIO_ROOT_PASSWORD:-minio123}"
LAKEHOUSE_BUCKET="${LAKEHOUSE_S3_BUCKET:-lakehouse}"

if [ "${MINIO_ENDPOINT}" = "http://minio:9000" ]; then
  MINIO_ENDPOINT="http://localhost:9000"
fi

docker run --rm --network host --entrypoint /bin/sh minio/mc -c "\
  mc alias set ${MINIO_ALIAS} ${MINIO_ENDPOINT} ${MINIO_USER} ${MINIO_PASSWORD} >/dev/null && \
  mc mb --ignore-existing ${MINIO_ALIAS}/${LAKEHOUSE_BUCKET} >/dev/null"

echo "Bucket ready: ${LAKEHOUSE_BUCKET}"
