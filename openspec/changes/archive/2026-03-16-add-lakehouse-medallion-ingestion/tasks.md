## 1. Lakehouse foundation in local Docker

- [x] 1.1 Add Docker services and bootstrap scripts for "Decision: Use MinIO + Spark + Iceberg + Nessie for local lakehouse".
- [x] [P] 1.2 Create local environment templates, credentials, and health checks for Lakehouse stack for local Docker environments.
- [x] 1.3 Add operational runbook for local start/stop/reset and storage initialization.

## 2. Bronze and Silver processing pipeline

- [x] 2.1 Implement Spark Bronze ingestion jobs for Bronze ingestion is raw and append-only with mandatory ingestion metadata fields.
- [x] [P] 2.2 Define and version Bronze schema contract files and source-to-bronze mapping configuration.
- [x] 2.3 Implement Spark Silver transformation jobs for Silver layer enforces schema and quality rules with fail-fast validation.
- [x] [P] 2.4 Add batch-level data quality checks and error reporting for Silver job runs.

## 3. Gold datasets and reproducible references

- [x] 3.1 Implement Gold materialization jobs for Gold layer provides training-ready datasets with deterministic transforms.
- [x] [P] 3.2 Add metadata publishing for snapshot lineage and table/version references.
- [x] 3.3 Implement API and DB contract updates for Training runs bind to immutable dataset references.
- [x] 3.4 Add run submission validation for lakehouse reference completeness and reproducibility.

## 4. Worker integration and migration safety

- [x] 4.1 Implement worker dataset resolution for Workers consume snapshot-bound Gold data.
- [x] [P] 4.2 Add compatibility path for "Decision: Introduce migration-compatible dataset resolution" with legacy file-based datasets.
- [x] 4.3 Add execution logs that persist catalog/table/reference/snapshot identifiers per run.
- [x] 4.4 Implement contract tests for "Decision: Bind pipeline runs to immutable data references" across API and workers.

## 5. Medallion governance, rollout, and verification

- [x] 5.1 Enforce medallion guardrails aligned with "Decision: Implement strict medallion boundaries" for Bronze/Silver/Gold ownership and write policies.
- [x] [P] 5.2 Add retention and compaction policy scripts plus rollback toggles for dual-path operation.
- [x] 5.3 Execute staged rollout checklist (foundation -> bronze -> silver -> gold -> worker cutover) in local Docker.
- [x] 5.4 Validate end-to-end reproducibility by running two trainings on the same snapshot and comparing metadata outputs.
