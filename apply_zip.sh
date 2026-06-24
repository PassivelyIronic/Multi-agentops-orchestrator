#!/usr/bin/env bash
# Usage: ./apply_zip.sh path/to/agentops-orchestrator.zip
#
# Safely merges a phase-delivery zip into the current repo directory,
# regardless of how deeply nested the zip's internal folder is (Windows
# duplicate-download "(2)" suffixes, double-nested extraction, etc.).
# It finds the directory that actually contains pyproject.toml inside the
# zip and copies *from there* — so "forgot one level of nesting" becomes
# impossible instead of a recurring manual guessing game.

set -euo pipefail

ZIP_PATH="${1:-}"
if [ -z "$ZIP_PATH" ]; then
  echo "Usage: $0 path/to/agentops-orchestrator.zip"
  exit 1
fi
if [ ! -f "$ZIP_PATH" ]; then
  echo "File not found: $ZIP_PATH"
  exit 1
fi

TMP_DIR=$(mktemp -d)
unzip -q "$ZIP_PATH" -d "$TMP_DIR"

PYPROJECT=$(find "$TMP_DIR" -name pyproject.toml -print -quit)
if [ -z "$PYPROJECT" ]; then
  echo "Could not find pyproject.toml anywhere inside the zip — aborting, nothing copied."
  rm -rf "$TMP_DIR"
  exit 1
fi
SRC_DIR=$(dirname "$PYPROJECT")

echo "Found project root inside zip at: $SRC_DIR"
echo "Copying its contents into: $(pwd)"
cp -r "$SRC_DIR"/. .

rm -rf "$TMP_DIR"
echo ""
echo "Done. Next: run 'git status' and verify NO .env / workspace/ / __pycache__ show up before 'git add'."
