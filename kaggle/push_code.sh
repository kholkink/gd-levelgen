#!/usr/bin/env bash
# Packages the code (gdgen + scripts + tests + kaggle + colab; no data, game files, mod or RE material) as the private Kaggle
# dataset "<user>/gd-levelgen-code". Kaggle kernels and the Colab worker take the code from there.
# Usage: bash kaggle/push_code.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/gd-levelgen/bin/python}
KAGGLE=${KAGGLE:-$(dirname "$PY")/kaggle}
USER_NAME=$($PY kaggle/kutil.py whoami)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
rsync -a --exclude '__pycache__' --exclude '.pytest_cache' --exclude 'core/build/' gdgen scripts tests kaggle colab README.md "$STAGE/"
VERSION=$(date -u +%Y%m%d%H%M%S)
echo "$VERSION" > "$STAGE/CODE_VERSION"
for must in gdgen/tokens.py gdgen/data/vocab_v1.json gdgen/data/objects_2.2081.json scripts/pretokenize.py scripts/train.py colab/worker.py colab/session_setup.sh gdgen/sim/core/gdsim.cpp gdgen/sim/fast.py; do
  [ -f "$STAGE/$must" ] || { echo "MISSING in staged code: $must"; exit 1; }
done
grep -rIl --exclude=push_code.sh "KGAT_\|sk-[A-Za-z0-9]\{30,\}\|hf_[A-Za-z0-9]\{30,\}" "$STAGE" && { echo "secret found in staged code"; exit 1; } || true
cat > "$STAGE/dataset-metadata.json" <<JSON
{"title": "gd-levelgen-code", "id": "${USER_NAME}/gd-levelgen-code", "licenses": [{"name": "other"}]}
JSON
if "$KAGGLE" datasets status "${USER_NAME}/gd-levelgen-code" >/dev/null 2>&1; then
  "$KAGGLE" datasets version -p "$STAGE" -m "code $VERSION" --dir-mode zip
else
  "$KAGGLE" datasets create -p "$STAGE" --dir-mode zip
fi
mkdir -p build && echo "$VERSION" > build/code_version.txt
echo "dataset: https://www.kaggle.com/datasets/${USER_NAME}/gd-levelgen-code (private), version $VERSION"
