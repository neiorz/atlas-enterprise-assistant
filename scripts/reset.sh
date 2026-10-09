#!/usr/bin/env bash
# FR-B8: wipe the vector index and rebuild it from data/docs/ in one command.
# Acceptance criterion: "bash scripts/reset.sh works".
set -euo pipefail

cd "$(dirname "$0")/.."

# Prefer the project venv, then python3, then python. Calling bare `python`
# fails on systems that only ship python3 (Ubuntu), and with set -e that used
# to abort *after* the old index was already deleted.
if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

echo "==> Using interpreter: $PY"

echo "==> Deleting local Qdrant index + policy index..."
rm -rf ./data/qdrant_index
rm -f ./data/policy_index.json

echo "==> Rebuilding corpus chunks + policy index..."
"$PY" -m src.ingest

echo "==> Rebuilding vector index (embeddings)..."
"$PY" -m src.vectorstore --force

echo "==> Done. Index reset complete."
