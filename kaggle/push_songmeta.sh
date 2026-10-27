#!/usr/bin/env bash
# Uploads data/songs/song_urls.jsonl (scripts/fetch_song_urls.py) as the private Kaggle dataset
# "<user>/gd-levelgen-songmeta"; the songs kernels download the audio from those URLs.
# Usage: bash kaggle/push_songmeta.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/gd-levelgen/bin/python}
KAGGLE=${KAGGLE:-$(dirname "$PY")/kaggle}
USER_NAME=$($PY kaggle/kutil.py whoami)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp data/songs/song_urls.jsonl "$STAGE/"
cat > "$STAGE/dataset-metadata.json" <<JSON
{"title": "gd-levelgen-songmeta", "id": "${USER_NAME}/gd-levelgen-songmeta", "licenses": [{"name": "other"}]}
JSON
if "$KAGGLE" datasets status "${USER_NAME}/gd-levelgen-songmeta" >/dev/null 2>&1; then
  "$KAGGLE" datasets version -p "$STAGE" -m "$(wc -l < data/songs/song_urls.jsonl) songs" --delete-old-versions
else
  "$KAGGLE" datasets create -p "$STAGE"
fi
echo "dataset: https://www.kaggle.com/datasets/${USER_NAME}/gd-levelgen-songmeta (private), $(wc -l < data/songs/song_urls.jsonl) songs"
