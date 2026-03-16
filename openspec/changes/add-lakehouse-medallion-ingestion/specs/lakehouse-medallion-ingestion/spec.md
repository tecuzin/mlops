## ADDED Requirements

### Requirement: Lakehouse stack for local Docker environments

The platform SHALL provide a local Docker lakehouse stack composed of object storage, processing engine, and table catalog services required for medallion ingestion workflows.

#### Scenario: Local stack bootstraps successfully

- **WHEN** an operator starts the platform with the lakehouse profile enabled
- **THEN** object storage, catalog, and Spark services are reachable and ready for batch jobs

### Requirement: Bronze ingestion is raw and append-only

The ingestion pipeline SHALL persist source records to Bronze tables without destructive updates, and each record MUST include ingestion metadata for traceability.

#### Scenario: New batch is ingested to Bronze

- **WHEN** a Spark Bronze ingestion job runs for a source dataset
- **THEN** new records are appended and each record contains source identifier, ingestion timestamp, and batch identifier

### Requirement: Silver layer enforces schema and quality rules

Silver transformations SHALL normalize Bronze records into typed, validated structures and MUST fail the job when mandatory quality constraints are violated.

#### Scenario: Invalid data is detected during Silver transformation

- **WHEN** a Bronze batch contains records violating required schema or nullability rules
- **THEN** the Silver job fails with a quality error and no invalid Silver output is committed

### Requirement: Gold layer provides training-ready datasets

Gold transformations SHALL materialize curated training datasets from Silver data using deterministic transformations and partitioning rules.

#### Scenario: Gold dataset is generated for training

- **WHEN** the Gold materialization job completes for a configured domain
- **THEN** a training-ready Gold table version is committed and queryable

### Requirement: Training runs bind to immutable dataset references

Run creation and persistence MUST support immutable lakehouse references including table identity and version pointer so that model training is reproducible.

#### Scenario: Run is created with a lakehouse dataset reference

- **WHEN** a client submits a run request that targets a Gold dataset
- **THEN** the system stores the table reference and snapshot/version identifier used for downstream worker execution

### Requirement: Workers consume snapshot-bound Gold data

Workers SHALL resolve lakehouse dataset references and read Gold data from the specified immutable snapshot before processing.

#### Scenario: Worker resolves dataset snapshot for execution

- **WHEN** a worker starts processing a run with a lakehouse dataset reference
- **THEN** the worker reads the exact referenced snapshot and logs dataset identity metadata for traceability
