# Security History Cleanup Runbook

This repository history was rewritten to remove tracked `.env` content.

## Maintainer actions

After review, push rewritten history:

```bash
git push --force-with-lease origin main
git push --force-with-lease origin security
```

If tags were rewritten, push updated tags:

```bash
git push --force-with-lease --tags
```

## Collaborator actions

Because history changed, each contributor must resync safely.

Recommended path:

```bash
# Option A: fresh clone
git clone <repo-url>
```

Alternative for existing local clones:

```bash
git fetch --all --prune
git checkout main
git reset --hard origin/main
git checkout security
git reset --hard origin/security
```

Then re-apply local work on top of new history with cherry-pick/rebase.

## Verification commands

Run these checks after force-push:

```bash
git rev-list --branches --tags | xargs -n 1 git grep -n "MISTRAL_API_KEY="
git rev-list --branches --tags | xargs -n 1 git grep -n "ROBodYzmKi4OguGvgmZMN7q8wUWAksg6"
```

Expected: no output.
