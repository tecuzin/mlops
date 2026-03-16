## Context

The current platform stores training and evaluation data as file paths and seeded records, which is operationally simple but weak for governance and reproducibility. The change introduces a local Docker-first lakehouse entry layer that remains compatible with existing FastAPI and worker orchestration patterns while improving data quality controls and versioned dataset consumption.

Key constraints:
- Local deployment only (no cloud dependency in this phase).
- Existing API and worker contracts must remain backward-compatible during migration.
- Data engineering flow must follow medallion semantics and Parquet-oriented processing.

## Goals / Non-Goals

**Goals:**

- Provide Bronze/Silver/Gold dataset layers with Spark transformations.
- Persist datasets as Iceberg tables backed by Parquet files.
- Enable immutable training dataset selection through table/version/snapshot metadata.
- Integrate lakehouse-managed datasets into API run creation and worker data resolution.

**Non-Goals:**

- Cloud-native catalog/storage integration (for example Glue or managed object stores).
- Streaming ingestion and near-real-time processing.
- Rewriting training algorithms or model architectures.

## Decisions

### Decision: Use MinIO + Spark + Iceberg + Nessie for local lakehouse

This stack provides local object storage, distributed transformation, table-level ACID behavior, and versioned catalog semantics in Docker. It balances governance needs and local operability.  
Alternative considered: Delta Lake-based setup; rejected for this phase to prioritize catalog-level branching/tagging with Nessie.

### Decision: Implement strict medallion boundaries

Bronze SHALL remain raw and append-only, Silver SHALL enforce schema and quality checks, and Gold SHALL expose training-ready curated datasets. This separation isolates data quality concerns and improves rollback/debugging.  
Alternative considered: single curated layer; rejected because it weakens lineage and replayability.

### Decision: Bind pipeline runs to immutable data references

Run creation SHALL store lakehouse identifiers (`catalog`, `namespace`, `table`, `reference`, `snapshot_id`) so workers consume deterministic dataset versions.  
Alternative considered: storing only table names; rejected due to non-deterministic retraining risk.

### Decision: Introduce migration-compatible dataset resolution

Workers SHALL support both legacy file-path datasets and new lakehouse snapshots during transition. This avoids disruptive cutover and supports phased adoption.

## Risks / Trade-offs

- [Operational complexity increase] -> Provide minimal default Compose profile and documented bootstrap scripts.
- [Schema drift at Bronze to Silver boundary] -> Enforce explicit schema contracts and fail-fast quality checks in Silver jobs.
- [Storage growth in Bronze] -> Add retention and compaction policy after initial MVP stabilization.
- [Dual path support increases code surface] -> Time-box migration window and remove legacy-only code once adoption reaches target.

## Migration Plan

1. Add Docker services and bootstrap scripts for MinIO, Nessie, and Spark.
2. Implement Bronze ingestion jobs and metadata conventions.
3. Implement Silver normalization and validation jobs.
4. Implement Gold dataset materialization for model training.
5. Extend API/DB schemas for dataset reference versioning.
6. Update workers to resolve and read snapshot-bound Gold datasets.
7. Run dual-path validation (legacy and lakehouse), then switch defaults to lakehouse.

Rollback strategy:
- Keep legacy file-path dataset flow active behind a feature flag during rollout.
- If lakehouse processing fails, create runs with legacy datasets and disable lakehouse selection in API/UI.

## Open Questions

- Which metadata fields are mandatory at run creation versus derived at execution time?
- Should security worker consume Gold directly or a dedicated security-oriented Silver view?
- What retention horizon is acceptable for Bronze in local Docker environments?
