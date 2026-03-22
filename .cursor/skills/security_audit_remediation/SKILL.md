# Security Audit and Remediation Playbook

This skill defines the standard workflow to audit and remediate secrets, weak credentials, and environment configuration issues in this repository.

## Objective

Eliminate exposed secrets, remove hardcoded credentials, enforce environment-driven configuration, and validate that end-to-end workflows still work after remediation.

## Scope

Applies to:
- `.env` and `.env.example`
- `.gitignore`
- `docker-compose.yml`
- CI security guardrails (`.gitleaks.toml`, SonarQube config, CI workflows)
- Git history and leak cleanup operations

## Non-Negotiable Rules

1. Never commit real secrets.
2. Revoke and rotate any leaked credential before claiming completion.
3. Use environment variables instead of hardcoded service credentials.
4. Keep `.env.example` complete and `.env` untracked.
5. Re-audit at the end (current tree + git history).
6. Re-test all critical workflows and lakehouse flows before closure.

## Workflow

### Phase 1 - First Deliverable: Skill/Runbook

Create or update a dedicated security runbook skill under `.cursor/skills/` before any remediation changes.

Minimum content:
- leak discovery commands
- file remediation checklist
- history rewrite procedure
- CI guardrails
- final re-audit and retest checklist

### Phase 2 - Secret Containment

1. Replace any exposed token in `.env` with an empty or placeholder value.
2. Ensure `.env` is ignored by git.
3. Align `.env` keys with `.env.example` keys (same variable set).
4. Confirm no obvious secret string remains in tracked files.

### Phase 3 - Configuration Hardening

1. Remove hardcoded database credentials from `docker-compose.yml`.
2. Reference `${POSTGRES_USER}`, `${POSTGRES_PASSWORD}`, `${POSTGRES_DB}` in service environments and connection strings.
3. Prefer non-trivial default values for local-only fallbacks.
4. Ensure MinIO and related credentials also use env-driven values.

### Phase 4 - Git History Cleanup

1. Rewrite git history to remove leaked tokens (for example with `git filter-repo` and `--replace-text`).
2. Expire reflog and run garbage collection after rewrite.
3. Verify leak signatures are no longer present in all revisions.
4. Coordinate force-push and re-clone instructions for collaborators.

### Phase 5 - Guardrails in CI

1. Add `.gitleaks.toml` tuned for this repo.
2. Add a CI workflow that runs:
   - gitleaks against current sources
   - gitleaks against git history
3. Add SonarQube project config and a CI scan workflow with quality gate wait.
4. Fail CI on secret findings or hardcoded credential patterns.

### Phase 6 - Final Validation (Mandatory)

Run all of the following:
1. Final security re-audit:
   - source scan
   - history scan
2. Docker/compose validation:
   - `docker compose config`
3. Functional workflow retests:
   - API dataset and run lifecycle checks
   - worker status transitions and logs
   - result persistence validation
4. Lakehouse retest:
   - bronze ingest
   - silver transform
   - gold materialization
   - metadata snapshot verification

## Definition of Done

A remediation campaign is done only if all conditions are true:
1. No real secret remains in tracked files.
2. Leaked secret signatures are removed from git history.
3. No hardcoded DB credentials remain in compose.
4. `.env` is untracked and `.env.example` is complete.
5. Gitleaks and SonarQube checks are configured in CI.
6. Final re-audit and workflow/lakehouse retests are green (or blocked with explicit evidence and mitigation).
