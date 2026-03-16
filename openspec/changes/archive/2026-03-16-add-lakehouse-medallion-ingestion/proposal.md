## Why

The current MLOps flow consumes training data from local files and database records, which limits traceability, reproducibility, and governed evolution of datasets. We need a local-first lakehouse entry layer to standardize ingestion and provide versioned, auditable training-ready data.

## What Changes

- Introduce a local Docker lakehouse stack based on MinIO, Spark, Iceberg, and Nessie.
- Define a medallion ingestion flow: Bronze (raw immutable), Silver (validated/normalized), Gold (training-ready).
- Standardize Spark pipelines that ingest source data and write Parquet-backed Iceberg tables.
- Add dataset version metadata (table, branch/tag, snapshot) so training runs can bind to immutable data snapshots.
- Align API and worker contracts to consume governed Gold datasets instead of ad-hoc file paths.

## Capabilities

### New Capabilities

- `lakehouse-medallion-ingestion`: Governed local data ingestion with Bronze/Silver/Gold layers, Spark transforms, and snapshot-based dataset consumption for model training.

### Modified Capabilities

(none)

## Non-objectives

- No cloud deployment target in this change (local Docker only).
- No real-time streaming ingestion; batch ingestion only.
- No replacement of model training logic itself, except input dataset resolution contracts.

## Impact

- Affected specs: `lakehouse-medallion-ingestion` (new capability).
- Affected code: `docker-compose.yml`, `db/models.py`, `api/schemas.py`, `api/main.py`, `db/init_db.py`, `workers/training/worker.py`, `workers/evaluation/worker.py`, `workers/security/worker.py`, new `lakehouse/` Spark jobs and configs.
- Dependencies/systems: MinIO object storage, Spark runtime, Iceberg table format, Nessie catalog, Parquet data layout.
