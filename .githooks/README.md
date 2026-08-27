# Repo git hooks

Activate (one-time, repo-local):

```bash
git config core.hooksPath .githooks
```

## pre-commit

Regenerates `env.yaml` (and stages `uv.lock` when it changed) whenever
`pyproject.toml` is in the staged diff. `env.yaml` is generated output
(`scripts/sync_env.py`); the hook keeps it from silently drifting from the
project's dependency list.

- Manual regeneration: `python scripts/sync_env.py` (or `--check` to test).
- CI check: `python scripts/sync_env.py --check` (exit 1 = stale).

See `scripts/sync_env.py` for the derivation rules: the dependency set comes
from `pyproject.toml`, the versions from `uv export` of `uv.lock` (full
resolved closure, torch included — the Databricks custom environment is built
from scratch, so nothing is preinstalled).
