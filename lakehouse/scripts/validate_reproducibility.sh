#!/usr/bin/env bash
set -euo pipefail

RUN_A_JSON="${1:-}"
RUN_B_JSON="${2:-}"

if [ -z "${RUN_A_JSON}" ] || [ -z "${RUN_B_JSON}" ]; then
  echo "Usage: $0 <run-a.json> <run-b.json>"
  exit 1
fi

SNAP_A="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("train_lakehouse_ref",{}).get("snapshot_id",""))' "${RUN_A_JSON}")"
SNAP_B="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("train_lakehouse_ref",{}).get("snapshot_id",""))' "${RUN_B_JSON}")"

if [ -z "${SNAP_A}" ] || [ -z "${SNAP_B}" ]; then
  echo "Missing snapshot_id in one of the run files."
  exit 1
fi

if [ "${SNAP_A}" != "${SNAP_B}" ]; then
  echo "Snapshot mismatch: ${SNAP_A} vs ${SNAP_B}"
  exit 1
fi

echo "Reproducibility check passed: both runs use snapshot ${SNAP_A}"
