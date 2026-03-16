# Lakehouse Rollout Checklist (Local Docker)

## Stage 1 - Foundation

- [ ] Start `minio`, `nessie`, `spark`
- [ ] Run `bootstrap_local.sh`
- [ ] Validate service health endpoints

## Stage 2 - Bronze

- [ ] Run `run_bronze.sh`
- [ ] Verify ingestion metadata columns
- [ ] Confirm append-only behavior across two batches

## Stage 3 - Silver

- [ ] Run `run_silver.sh`
- [ ] Validate schema and quality checks
- [ ] Confirm failing input triggers fail-fast behavior

## Stage 4 - Gold

- [ ] Run `run_gold.sh`
- [ ] Validate snapshot metadata file output
- [ ] Validate metadata includes `catalog`, `reference`, and `catalog_commit_id`
- [ ] Confirm training-ready export paths are present

## Stage 5 - API and Worker Cutover

- [ ] Create run payload with `train_lakehouse_ref` / `eval_lakehouse_ref`
- [ ] Confirm workers resolve snapshot-bound paths
- [ ] Verify fallback works when `LAKEHOUSE_ENABLED=false`
- [ ] Verify mismatch on `catalog/reference/snapshot_id` fails loudly before processing
