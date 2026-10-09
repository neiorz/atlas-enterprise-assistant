#!/usr/bin/env bash
# FR-B8: wipe the vector index and rebuild it from data/docs/ in one command.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Deleting local Qdrant index + policy index..."
rm -rf ./data/qdrant_index
rm -f ./data/policy_index.json

echo "==> Rebuilding corpus chunks + policy index..."
python -m src.ingest

echo "==> Rebuilding vector index (embeddings)..."
python -m src.vectorstore --force

echo "==> Done. Index reset complete."
