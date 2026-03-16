## 1. Catalog wiring foundation

- [x] 1.1 Implement **Configure Spark with an explicit Nessie catalog for medallion jobs** in `lakehouse/jobs/bronze_ingest.py`, `lakehouse/jobs/silver_transform.py`, `lakehouse/jobs/gold_materialize.py`, and launcher scripts.
- [x] [P] 1.2 Update `docker-compose.yml` and `.env.example` to enforce catalog defaults needed by **Lakehouse stack for local Docker environments**.
- [x] [P] 1.3 Add runbook checks in `lakehouse/RUNBOOK.md` to verify catalog reachability and effective table publication.

## 2. Snapshot reference contracts

- [x] 2.1 Implement **Derive run snapshot references from catalog commit/snapshot metadata** in Gold metadata emission and API contract validation.
- [x] 2.2 Update `workers/lakehouse_ref.py` to satisfy **Snapshot references are validated before worker consumption** with explicit mismatch errors.
- [x] [P] 2.3 Update worker logs (`workers/training/worker.py`, `workers/evaluation/worker.py`, `workers/security/worker.py`) to expose catalog identity and snapshot traceability.

## 3. Worker compatibility and fallback

- [x] 3.1 Implement **Keep metadata-export fallback for worker data access** while preserving deterministic provenance from catalog-bound snapshots.
- [x] 3.2 Ensure **Operational fallback remains available for legacy execution** using `LAKEHOUSE_ENABLED` toggling and documented rollback behavior.

## 4. Verification and regression safety

- [x] [P] 4.1 Extend/adjust lakehouse tests to validate **Medallion jobs publish tables through Nessie catalog** and snapshot reproducibility.
- [x] [P] 4.2 Add contract tests that validate **Lakehouse stack for local Docker environments** now includes active catalog usage, not only service presence.
- [x] 4.3 Execute test suite subsets and document results for no-regression on legacy dataset fallback paths.

## 5. Documentation updates

- [x] [P] 5.1 Update `README.md` to describe how Nessie is actively used (catalog wiring, snapshot semantics, and fallback behavior).
- [x] [P] 5.2 Update `lakehouse/RUNBOOK.md` and `lakehouse/ROLLOUT_CHECKLIST.md` with operational checks for catalog-backed publication and rollback steps.
- [x] [P] 5.3 Update `docs/uml_schemas.md` to reflect catalog-backed flow between Spark jobs, metadata, and workers.
