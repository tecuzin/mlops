# Docker Build Optimization Playbook

This file defines the standard workflow for future Docker build optimization campaigns in this repository.

## Objective

Reduce Docker build and rebuild time without breaking runtime behavior or image correctness.

## Scope

Applies to:
- `docker/Dockerfile.*`
- `docker-compose.yml`
- `.dockerignore`
- Docker dependency files under `docker/requirements/`

## Non-Negotiable Rules

1. Measure before and after every change.
2. Keep functional parity (same service behavior after optimization).
3. No destructive git operations.
4. Do not introduce secrets in Dockerfiles or build args.
5. Stop and report if unexpected unrelated workspace changes appear.

## Campaign Workflow

### Phase 1 - Baseline

Run and record:
1. Cold-ish build:
   - `DOCKER_BUILDKIT=1 /usr/bin/time -p docker compose build api ui mlflow training evaluation security`
2. Warm cache rebuild:
   - same command again
3. Optional per-image timing:
   - `DOCKER_BUILDKIT=1 /usr/bin/time -p docker compose build <service>`

Capture:
- total `real/user/sys`
- slowest build steps (`pip install`, `apt-get`, large `COPY`)
- cache hit ratio clues (`CACHED` count in logs)

### Phase 2 - Diagnosis

Check for common bottlenecks:
- dependency install done before stable inputs are isolated
- `COPY` order invalidates cache too often
- oversized build context (`.dockerignore` missing entries)
- duplicate dependencies across services
- expensive export/load behavior from builder strategy

### Phase 3 - Optimization

Apply changes in small batches:
1. Layer ordering for cache stability.
2. Requirements split (common vs service-specific).
3. BuildKit cache mounts:
   - pip: `--mount=type=cache,target=/root/.cache/pip`
   - apt: `--mount=type=cache,target=/var/cache/apt` and `/var/lib/apt`
4. Remove unnecessary `COPY` sources when already mounted at runtime.
5. Tighten `.dockerignore`.

### Phase 4 - Validation

Mandatory checks after optimization:
1. Compose syntax:
   - `docker compose config`
2. Full rebuild benchmark (run1/run2):
   - same command as baseline
3. Smoke runtime:
   - `docker compose up -d db api mlflow ui training evaluation security`
   - `GET /docs` on API
   - UI reachable
   - MLflow reachable
   - representative API endpoints return expected status
4. Logs sanity:
   - check API and worker logs for startup/runtime errors
5. Clean shutdown:
   - `docker compose down`

## Buildx Guidance

Use `buildx` only when it provides clear value:
- CI remote cache sharing (`cache-to/cache-from` with registry backend)
- multi-platform builds

Local caveat:
- exporting local cache and loading huge images can offset benefits.
- avoid parallel targets writing to the same local cache destination.

## Definition of Done

A campaign is DONE only if all are true:
1. Warm rebuild time improved or remains stable for target services.
2. No regression in smoke tests.
3. `docker compose config` passes.
4. Changes are documented with measured numbers.

## Required Report Format

For each campaign, provide:
1. Context
   - branch, date, objective
2. Changes
   - files touched
   - optimization techniques applied
3. Results
   - baseline cold/warm
   - final cold/warm
   - delta seconds, percent gain, speedup factor
4. Validation
   - smoke checks status
   - known limitations
5. Decision
   - GO or NO-GO
   - next actions

## Quick Command Set

```bash
# Benchmark full stack build (twice)
DOCKER_BUILDKIT=1 /usr/bin/time -p docker compose build api ui mlflow training evaluation security
DOCKER_BUILDKIT=1 /usr/bin/time -p docker compose build api ui mlflow training evaluation security

# Runtime smoke
docker compose up -d db api mlflow ui training evaluation security
curl -fsS http://localhost:8000/docs >/dev/null
curl -fsS http://localhost:8501 >/dev/null
curl -fsS http://localhost:5001 >/dev/null
docker compose down
```
