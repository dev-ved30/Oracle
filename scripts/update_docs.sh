#!/usr/bin/env bash
# =====================================================================
# update_docs.sh — Clean, rebuild, and optionally push Sphinx docs
# =====================================================================

# Exit immediately on error
set -e

# Paths — customize as needed
SRC_DIR="src/oracle"
DOCS_SOURCE="docs/source"
DOCS_BUILD="docs/build"
HTML_DIR="$DOCS_BUILD/html"

echo "📚 [1/5] Cleaning old documentation..."
rm -rf "$DOCS_BUILD" "$DOCS_SOURCE"/_autosummary

echo "🧰 [2/5] Regenerating API docs from $SRC_DIR..."
sphinx-apidoc -f -o "$DOCS_SOURCE" "$SRC_DIR" --separate

echo "🧼 [3/5] Removing old .pyc cache (optional)..."
find "$SRC_DIR" -name "__pycache__" -type d -exec rm -rf {} +

echo "🏗️ [4/5] Building HTML documentation..."
make -C docs clean
make -C docs html

echo "✅ Documentation built at: $HTML_DIR"