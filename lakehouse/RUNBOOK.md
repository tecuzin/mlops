# Lakehouse Local Runbook

## Scope

This runbook covers local Docker operations for the lakehouse entry stack:
- MinIO (object storage)
- Nessie (catalog/versioning)
- Spark (batch medallion jobs)

## Start

```bash
docker compose up -d minio nessie spark
```

## Health Checks

```bash
curl -f http://localhost:9000/minio/health/live
curl -f http://localhost:19120/q/health
```

Spark UI:
- [http://localhost:8081](http://localhost:8081)

MinIO Console:
- [http://localhost:9001](http://localhost:9001)

## Bootstrap bucket

```bash
./lakehouse/scripts/bootstrap_local.sh
```

## Run medallion jobs

```bash
./lakehouse/scripts/run_bronze.sh
./lakehouse/scripts/run_silver.sh
./lakehouse/scripts/run_gold.sh
```

## Verify catalog-backed publication

After running medallion jobs, confirm metadata contains explicit catalog identity:

```bash
python -m json.tool lakehouse/metadata/gold_rag_qa_train_ready.json
```

Expected keys:
- `catalog` = `nessie`
- `table` (for example `gold.rag_qa_train_ready`)
- `reference` (for example `main`)
- `snapshot_id`
- `catalog_commit_id`

## Reset local lakehouse state

```bash
docker compose stop spark nessie minio
docker volume rm mlops_lakehouse_data
docker compose up -d minio nessie spark
```

## Rollback toggle

Set `LAKEHOUSE_ENABLED=false` for workers to fallback to legacy file-path datasets.
