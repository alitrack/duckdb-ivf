#!/usr/bin/env bash
# Ensure a python interpreter with the `duckdb` package, then run the SQL e2e.
#
# Resolution order:
#   1. `python3` (already has `duckdb` — e.g. CI after `pip install duckdb`)
#   2. repo-local `.venv/bin/python` (uv-managed; created once per checkout)
# Usage: bash tests/run_host.sh <path/to/vss.duckdb_extension>
set -euo pipefail

ARTIFACT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY=""
if python3 -c 'import duckdb' >/dev/null 2>&1; then
  PY=python3
elif [ -x "$REPO_ROOT/.venv/bin/python" ] && "$REPO_ROOT/.venv/bin/python" -c 'import duckdb' >/dev/null 2>&1; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  echo "error: no python with the \`duckdb\` package found (checked python3 and $REPO_ROOT/.venv)." >&2
  echo "  create one (uv, per our env convention):" >&2
  echo "    uv venv $REPO_ROOT/.venv --python 3.11 && \\" >&2
  echo "    uv pip install --python $REPO_ROOT/.venv/bin/python duckdb" >&2
  exit 1
fi

echo "e2e host: $PY ($( "$PY" -c 'import duckdb,sys; print(duckdb.__version__)' ))"
exec "$PY" "$REPO_ROOT/tests/run_sql_tests.py" "$ARTIFACT"
