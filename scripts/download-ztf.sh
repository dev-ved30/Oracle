#!/usr/bin/env bash
set -euo pipefail

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown is required. Install it with: python -m pip install gdown" >&2
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="$repo_root/data/BTSv3"
mkdir -p "$data_dir"

gdown "https://drive.google.com/drive/folders/1g7KBbTqmSHshTd3u-hruWfALEzJ6bvqi" \
    -O "$data_dir" --continue
