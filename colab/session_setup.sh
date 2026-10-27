#!/usr/bin/env bash
# Once per Colab session (run by colab/worker.py). Idempotent: a step is skipped when its result is already there.
#   1. CUDA check with Colab's preinstalled torch (nothing else to install: numpy + torch are all the trainer needs)
#   2. pretokenized levels: outputs of the Kaggle tok kernels (kaggle/kernel.py push tok) -> $GD_TOKENS/part_*/
# Env from the worker: GD_RUNS, GD_TOKENS, GD_PY, GD_TOK_STREAMS. GD_TOK_KERNELS (space-separated refs) overrides the kernel lookup.
set -euo pipefail
cd "$(dirname "$0")/.."
t0=$(date +%s)
say() { echo "[setup +$(( $(date +%s) - t0 ))s] $*"; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
say "start: $(nproc) CPUs, $(free -g | awk '/Mem/{print $2}') GB RAM, $(df -h /content | awk 'NR==2{print $4}') free disk"
"$GD_PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "no CUDA: switch the runtime to a GPU (H100 / A100 / RTX 6000)"
print("device", torch.cuda.get_device_name(), "capability", torch.cuda.get_device_capability())
PY
mkdir -p "$GD_RUNS" "$GD_TOKENS"
if [ "${GD_SKIP_TOKENS:-0}" = 1 ]; then
  say "no queued job reads \$GD_TOKENS: the pretokenized levels are not downloaded"
else

# Kaggle's search is fuzzy (it also returns e.g. gd-levelgen-eval): keep exactly gd-levelgen-tok-<i>of<n>
KERNELS=${GD_TOK_KERNELS:-$(kaggle kernels list --mine -s gd-levelgen-tok --csv --page-size 100 | tail -n +2 | cut -d, -f1 \
  | grep -E '/gd-levelgen-tok-[0-9]+of[0-9]+$' || true)}
[ -n "$KERNELS" ] || { echo "no gd-levelgen-tok-* kernels found"; exit 1; }
for k in $KERNELS; do
  name=$(basename "$k")
  streams=$(echo "${GD_TOK_STREAMS:-layout,deco}" | tr "," "|")
  if [ -f "$GD_TOKENS/.ready-$name-${streams//|/+}" ]; then
    say "$name present"
    continue
  fi
  say "downloading the output of $k"
  tmp="/content/tok_download/$name"
  rm -rf "$tmp" && mkdir -p "$tmp"
  # only the requested streams' shards (GD_TOK_STREAMS, comma-separated; default layout,deco) + index and vocab
  streams=$(echo "${GD_TOK_STREAMS:-layout,deco}" | tr ',' '|')
  kaggle kernels output "$k" -p "$tmp" -q --file-pattern "(($streams)_[0-9]+\.bin|index\.jsonl|vocab\.json)\$"
  found=0
  for part in "$tmp"/tokens/*/; do
    [ -f "$part/index.jsonl" ] || continue
    rm -rf "$GD_TOKENS/$(basename "$part")"
    mv "$part" "$GD_TOKENS/"
    found=1
  done
  [ "$found" = 1 ] || { echo "$k has no tokens/*/index.jsonl in its output: $(ls "$tmp")"; exit 1; }
  rm -rf "$tmp"
  touch "$GD_TOKENS/.ready-$name-${streams//|/+}"
done
say "tokens: $(cat "$GD_TOKENS"/*/index.jsonl | wc -l) levels in $(ls -d "$GD_TOKENS"/*/ | wc -l) parts, $(du -sh "$GD_TOKENS" | cut -f1)"
fi

if [ -n "${GD_SETUP_ARTIFACTS:-}" ]; then
  python colab/artifacts.py restore $GD_SETUP_ARTIFACTS
fi
say "ready: $(df -h /content | awk 'NR==2{print $4}') free disk"
