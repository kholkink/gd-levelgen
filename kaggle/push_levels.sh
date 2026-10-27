#!/usr/bin/env bash
# Uploads a consistent snapshot of the level database as the private Kaggle dataset "<user>/gd-levelgen-levels"
# (a new version when it exists). The tok kernels tokenize it on Kaggle.
# Usage: bash kaggle/push_levels.sh data/snapshots/levels_YYYYMMDD.sqlite
#   make the snapshot while the crawler runs (sqlite):  VACUUM INTO 'data/snapshots/levels_YYYYMMDD.sqlite'
# The snapshot is moved into data/snapshots/levels_YYYYMMDD/ (the upload folder) and stays there.
set -euo pipefail
cd "$(dirname "$0")/.."
SNAP=${1:?usage: push_levels.sh <snapshot.sqlite>}
PY=${PY:-$HOME/.venvs/gd-levelgen/bin/python}
KAGGLE=${KAGGLE:-$(dirname "$PY")/kaggle}
USER_NAME=$($PY kaggle/kutil.py whoami)
[ ! -e "$SNAP-journal" ] || { echo "$SNAP-journal exists: the snapshot is still being written"; exit 1; }
STAGE="${SNAP%.sqlite}"
mkdir -p "$STAGE"
[ -f "$SNAP" ] && mv "$SNAP" "$STAGE/"
cat > "$STAGE/dataset-metadata.json" <<JSON
{"title": "gd-levelgen-levels", "id": "${USER_NAME}/gd-levelgen-levels", "licenses": [{"name": "other"}]}
JSON
if "$KAGGLE" datasets status "${USER_NAME}/gd-levelgen-levels" >/dev/null 2>&1; then
  "$KAGGLE" datasets version -p "$STAGE" -m "$(basename "$STAGE")" --delete-old-versions
else
  "$KAGGLE" datasets create -p "$STAGE"
fi
echo "dataset: https://www.kaggle.com/datasets/${USER_NAME}/gd-levelgen-levels (private)"
