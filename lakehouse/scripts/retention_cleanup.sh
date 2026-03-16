#!/usr/bin/env bash
set -euo pipefail

BRONZE_ROOT="${BRONZE_ROOT:-./lakehouse/warehouse/bronze}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"

if [ ! -d "${BRONZE_ROOT}" ]; then
  echo "No bronze directory found at ${BRONZE_ROOT}, skipping."
  exit 0
fi

find "${BRONZE_ROOT}" -type f -mtime +"${RETENTION_DAYS}" -print -delete
echo "Retention cleanup completed for ${BRONZE_ROOT} (>${RETENTION_DAYS} days)."
