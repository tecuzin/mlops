# Medallion Guardrails (Local Docker)

## Bronze

- Bronze data is append-only and immutable.
- Bronze records MUST include `ingestion_ts`, `source_id`, and `batch_id`.
- No destructive update is allowed on Bronze paths.

## Silver

- Silver enforces schema and quality contracts.
- Invalid records MUST fail the transformation job.
- Silver tables are the canonical clean layer for curation and governance checks.

## Gold

- Gold datasets are deterministic and training-ready.
- Gold writes MUST publish snapshot metadata for reproducible run binding.
- Gold is the only authorized layer for model training inputs.

## Ownership and write policy

- Only Spark jobs in `lakehouse/jobs` can write Bronze/Silver/Gold.
- API and workers are read-only consumers for lakehouse datasets.
- Rollback to legacy datasets is controlled with `LAKEHOUSE_ENABLED=false`.
