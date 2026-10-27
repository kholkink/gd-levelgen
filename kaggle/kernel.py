#!/usr/bin/env python
"""Builds, pushes and follows the project's Kaggle kernels (heavy work runs there, not on the local CPU).

    python kaggle/kernel.py push tok --part 0/2      # tests + pretokenize part 0 of 2 (CPU)
    python kaggle/kernel.py push smoke               # tests + a short training run on the tok outputs (GPU T4)
    python kaggle/kernel.py push eval --eval-gen gen # generated levels of kernel gen vs featured levels (CPU)
    python kaggle/kernel.py push diag                # what kills the player on eval's featured levels (CPU)
    python kaggle/kernel.py status tok-0of2
    python kaggle/kernel.py logs smoke               # tail of the run log + per-step exit codes
    python kaggle/kernel.py output smoke             # download outputs to build/kernels/<slug>/out

Kernels read the code from the private dataset <user>/gd-levelgen-code (bash kaggle/push_code.sh first) and the
levels from <user>/gd-levelgen-levels (bash kaggle/push_levels.sh). Outputs of tok kernels are the input of smoke.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "kaggle"))
from kutil import kaggle_bin, username  # noqa: E402

PREFIX = "gd-levelgen-"

SETUP = r'''
import glob, json, os, shutil, subprocess, sys, time
from pathlib import Path

def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return Path(hits[0]) if hits else None

CODE = find("gdgen/tokens.py").parent.parent
SRC = Path("/tmp/src")                       # writable copy (pycache, pytest), outside /kaggle/working
shutil.copytree(CODE, SRC, dirs_exist_ok=True)
os.chdir(SRC)
W = Path("/kaggle/working")
ENV = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=str(SRC))
db = find("levels_*.sqlite")
if db:
    ENV["LEVELS_DB"] = str(db)
song_urls = find("song_urls.jsonl")
if song_urls:
    ENV["SONG_URLS"] = str(song_urls)
tok = sorted({Path(p).parent.parent for p in glob.glob("/kaggle/input/**/tokens/*/index.jsonl", recursive=True)})
if tok:
    link = Path("/tmp/tokens")                # every tok kernel output becomes a part of one dataset
    link.mkdir(exist_ok=True)
    for t in tok:
        for part in t.iterdir():
            if (part / "index.jsonl").exists() and not (link / part.name).exists():
                (link / part.name).symlink_to(part)
    ENV["TOKENS"] = str(link)
print("code", (SRC / "CODE_VERSION").read_text().strip(), "| db", db, "| tokens", tok, flush=True)
subprocess.run("nproc; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true; "
               "python -c 'import torch, numpy; print(\"torch\", torch.__version__, \"numpy\", numpy.__version__)'",
               shell=True)

RESULTS = {}
def step(name, cmd):
    print(f"\n===== {name}: {cmd}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, shell=True, env=ENV, cwd=SRC).returncode
    RESULTS[name] = {"rc": rc, "seconds": round(time.time() - t0)}
    print(f"===== {name}: exit {rc} after {time.time() - t0:.0f}s", flush=True)
    (W / "results.json").write_text(json.dumps(RESULTS, indent=1))
    return rc
'''


def tok_steps(part: str) -> str:
    i, n = part.split("/")
    return f'''
step("tests", "python -m pytest -q -p no:cacheprovider tests")
step("pretokenize", "python scripts/pretokenize.py --db $LEVELS_DB --part {part} --workers 4 "
                    "--out /kaggle/working/tokens/part_{i}of{n}")
'''


def smoke_steps(args) -> str:
    return f'''
step("tests", "python -m pytest -q -p no:cacheprovider tests/test_model.py tests/test_tokens.py")
step("generate-random", "python scripts/generate.py --random-init --out /kaggle/working/gen/random "
                        "--max-layout-tokens 3000 --max-deco-window-tokens 500 --max-deco-windows 20")
step("train", "python scripts/train.py --data $TOKENS --out /kaggle/working/runs/smoke --preset {args.preset} "
              "--ctx {args.ctx} --batch {args.batch} --accum {args.accum} --steps {args.steps} --warmup 50 "
              "--lr 6e-4 --log-every 10 --eval-every 100 --eval-windows 16 --save-every {args.steps} --workers 3 "
              "--grad-ckpt")
step("generate", "python scripts/generate.py --ckpt /kaggle/working/runs/smoke/ckpt.pt --out /kaggle/working/gen/smoke "
                 "--max-layout-tokens 6000 --max-deco-window-tokens 1500 --max-deco-windows 60 --temperature 0.9")
'''


def songs_steps(args) -> str:
    i, n = args.part.split("/")
    limit = f" --limit {args.limit}" if args.limit else ""
    return f'''
step("deps", "pip install -q 'soundfile>=0.12.1' && python -c 'import soundfile; print(soundfile.__libsndfile_version__)'")
step("song_features", "python scripts/song_features.py --urls $SONG_URLS --part {args.part} --workers 4 "
                      "--out /kaggle/working/song_features/part_{i}of{n}{limit}")
'''


def gen_steps(args) -> str:
    """Generation with the checkpoint found in the attached kernel output (e.g. the smoke run), several seeds."""
    seeds = " ".join(str(s) for s in range(args.samples))
    extra = f" --select-segments {args.select} --select-sim-seconds 20" if args.select else ""
    return f'''
ckpt = find("runs/*/ckpt.pt") or find("ckpt.pt")
print("checkpoint", ckpt, flush=True)
for seed in "{seeds}".split():
    step(f"generate-{{seed}}", f"python scripts/generate.py --ckpt {{ckpt}} --out /kaggle/working/gen/s{{seed}} --seed {{seed}} "
                             "--max-layout-tokens {args.max_layout_tokens} --max-deco-windows {args.max_deco_windows} "
                             "--max-deco-window-tokens {args.max_deco_window_tokens} --temperature {args.temperature}{extra}")
'''


def eval_steps(args) -> str:
    """Metrics + simulation of the generated levels in the attached kernel outputs against featured levels."""
    def spec(k: str) -> str:
        if k.startswith("ds:"):                  # a dataset (e.g. a Colab run artifact): ds:gdgen-runs-m1-gen
            slug = k[3:]
            return f"--gen {slug.removeprefix('gdgen-runs-')}='/kaggle/input/**/{slug}/**/level.txt'"
        return f"--gen {k}='/kaggle/input/**/{slug_of(k)}/**/gen/*/level.txt'"
    gens = " ".join(spec(k) for k in args.eval_gen.split(","))
    return f'''
step("evaluate", "python scripts/evaluate.py {gens} --ref-db $LEVELS_DB --ref-n {args.ref_n} --simulate "
                 "--sim-seconds {args.sim_seconds} --workers 4 --out /kaggle/working/eval")
'''


def diag_steps(args) -> str:
    """What kills the player on the reference levels of the attached eval output (whole level, then gameplay only)."""
    return f'''
metrics = find("eval/metrics.jsonl")
print("metrics", metrics, flush=True)
step("diag-full", f"python scripts/diag_sim.py --db $LEVELS_DB --metrics {{metrics}} --out /kaggle/working/diag/full "
                  "--seconds {args.sim_seconds} --workers 4")
step("diag-gameplay", f"python scripts/diag_sim.py --db $LEVELS_DB --metrics {{metrics}} --out /kaggle/working/diag/gameplay "
                      "--gameplay-only --seconds {args.sim_seconds} --workers 4")
'''


def build(job: str, args) -> tuple[str, Path, dict]:
    user = username()
    if job == "tok":
        i, n = args.part.split("/")
        slug = f"{PREFIX}tok-{i}of{n}"
        body, gpu = tok_steps(args.part), None
        datasets, kernels = [f"{user}/gd-levelgen-code", f"{user}/gd-levelgen-levels"], []
    elif job == "smoke":
        slug = f"{PREFIX}smoke"
        body, gpu = smoke_steps(args), args.accelerator
        datasets = [f"{user}/gd-levelgen-code"]
        kernels = [f"{user}/{PREFIX}tok-{i}of{args.tok_parts}" for i in range(args.tok_parts)]
    elif job == "gen":
        slug = f"{PREFIX}gen"
        body, gpu = gen_steps(args), args.accelerator
        datasets = [f"{user}/gd-levelgen-code"] + ([f"{user}/{args.ckpt_dataset}"] if args.ckpt_dataset else [])
        kernels = [] if args.ckpt_dataset else [f"{user}/{slug_of(args.ckpt_kernel)}"]
    elif job == "eval":
        slug = f"{PREFIX}eval"
        body, gpu = eval_steps(args), None
        names = args.eval_gen.split(",")
        datasets = [f"{user}/gd-levelgen-code", f"{user}/gd-levelgen-levels"] + \
            [f"{user}/{k[3:]}" for k in names if k.startswith("ds:")]
        kernels = [f"{user}/{slug_of(k)}" for k in names if not k.startswith("ds:")]
    elif job == "diag":
        slug = f"{PREFIX}diag"
        body, gpu = diag_steps(args), None
        datasets = [f"{user}/gd-levelgen-code", f"{user}/gd-levelgen-levels"]
        kernels = [f"{user}/{PREFIX}eval"]
    elif job == "songs":
        i, n = args.part.split("/")
        slug = f"{PREFIX}songs-{i}of{n}"
        body, gpu = songs_steps(args), None
        datasets, kernels = [f"{user}/gd-levelgen-code", f"{user}/gd-levelgen-songmeta"], []
    elif job == "tests":
        slug = f"{PREFIX}tests"
        body, gpu = 'step("tests", "python -m pytest -q -p no:cacheprovider -rf tests")\n', None
        datasets, kernels = [f"{user}/gd-levelgen-code"], []
    else:
        raise SystemExit(f"unknown job {job}")
    out = REPO / "build" / "kernels" / slug
    out.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": c.strip("\n").splitlines(keepends=True)} for c in (SETUP, body)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": bool(gpu), "enable_tpu": False,
            "enable_internet": job == "songs", "dataset_sources": datasets, "kernel_sources": kernels,
            "competition_sources": [], "model_sources": []}
    if gpu:
        meta["machine_shape"] = gpu
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return slug, out, meta


def slug_of(name: str) -> str:
    return name if name.startswith(PREFIX) else PREFIX + name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["build", "push", "status", "logs", "output"])
    ap.add_argument("job")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--tok-parts", type=int, default=2)
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    ap.add_argument("--preset", default="s")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ckpt-kernel", default="smoke", help="gen: kernel whose output holds runs/*/ckpt.pt")
    ap.add_argument("--ckpt-dataset", default="", help="gen: dataset holding ckpt.pt instead (e.g. gdgen-runs-m1)")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-layout-tokens", type=int, default=20000)
    ap.add_argument("--max-deco-windows", type=int, default=40)
    ap.add_argument("--max-deco-window-tokens", type=int, default=8000)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--select", type=int, default=0, help="gen: simulator selection every N segments (0: off)")
    ap.add_argument("--limit", type=int, default=0, help="songs: only the first N songs of the part (tests)")
    ap.add_argument("--eval-gen", default="gen",
                    help="eval: comma-separated kernels whose outputs hold gen/*/level.txt, or ds:<dataset>")
    ap.add_argument("--ref-n", type=int, default=300, help="eval: featured levels in the reference sample")
    ap.add_argument("--sim-seconds", type=float, default=60, help="eval: simulator budget per level")
    ap.add_argument("--timeout", type=int, default=6 * 3600, help="seconds (quota guard)")
    args = ap.parse_args()
    kg = kaggle_bin()
    if args.action in ("build", "push"):
        slug, out, meta = build(args.job, args)
        print(f"built {out}: datasets {meta['dataset_sources']}, kernels {meta['kernel_sources']}, "
              f"gpu {meta.get('machine_shape')}")
        if args.action == "push":
            cmd = [kg, "kernels", "push", "-p", str(out), "-t", str(args.timeout)]
            if meta.get("machine_shape"):
                cmd += ["--accelerator", meta["machine_shape"]]
            r = subprocess.run(cmd, capture_output=True, text=True)
            print((r.stdout + r.stderr).strip())
            return r.returncode
        return 0
    ref = f"{username()}/{slug_of(args.job)}"
    if args.action == "status":
        return subprocess.run([kg, "kernels", "status", ref]).returncode
    if args.action == "logs":
        return subprocess.run([kg, "kernels", "logs", ref]).returncode
    dest = REPO / "build" / "kernels" / slug_of(args.job) / "out"
    dest.mkdir(parents=True, exist_ok=True)
    return subprocess.run([kg, "kernels", "output", ref, "-p", str(dest)]).returncode


if __name__ == "__main__":
    sys.exit(main())
