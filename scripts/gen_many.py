"""Several scripts/generate.py runs in parallel on one GPU (distinct seeds), progress lines prefixed with the seed.

Generation is bound by the per-token Python work (grammar masks, sampling), not by the GPU, so N processes give
about N times the throughput until the CPU or GPU saturates.

    python scripts/gen_many.py --ckpt runs/m1/ckpt.pt --out gen/m1 --samples 8 [--parallel 8] [--first-seed 0] \
        -- --max-layout-tokens 60000 --max-deco-windows 60

Writes <out>/s<seed>/ (generate.py output, log.txt), <out>/summary.jsonl (one line per finished sample) and
<out>/ready-s<seed>-gameplay.txt, ready-s<seed>.txt: level strings as soon as the layout / the whole level is
done, at the top level so a run artifact keeps them as single files (scripts/watch_levels.py fetches them).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_KEYS = ("layout_finished", "layout_tokens", "deco_tokens", "gameplay_objects", "deco_objects",
                "layout_decode_errors", "deco_decode_errors", "seconds")


def pump(seed: int, proc: subprocess.Popen, log_path: Path, lock: threading.Lock) -> None:
    """Copies a run's output to its log; progress and error lines also go to stdout (the job log)."""
    with open(log_path, "w") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if "layout:" in line or "deco:" in line or "Error" in line or "Traceback" in line:
                with lock:
                    print(f"[s{seed}] {line.rstrip()}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- then arguments passed to every generate.py")
    args = ap.parse_args()
    extra = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.first_seed, args.first_seed + args.samples))
    lock, running, t0 = threading.Lock(), {}, time.time()
    with open(out / "summary.jsonl", "a") as summary:
        while seeds or running:
            while seeds and len(running) < args.parallel:
                seed = seeds.pop(0)
                d = out / f"s{seed}"
                d.mkdir(exist_ok=True)
                cmd = [sys.executable, "-u", str(ROOT / "scripts" / "generate.py"), "--ckpt", args.ckpt,
                       "--out", str(d), "--seed", str(seed), "--ready-dir", str(out),
                       "--ready-name", f"ready-s{seed}", *extra]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
                th = threading.Thread(target=pump, args=(seed, proc, d / "log.txt", lock), daemon=True)
                th.start()
                running[seed] = (proc, th)
            for seed, (proc, th) in list(running.items()):
                if proc.poll() is None:
                    continue
                th.join()
                del running[seed]
                rep = out / f"s{seed}" / "report.json"
                rec = {"seed": seed, "rc": proc.returncode}
                if rep.exists():
                    r = json.loads(rep.read_text())
                    rec |= {k: r.get(k) for k in SUMMARY_KEYS}
                    sim = r.get("simulation") or {}
                    rec["sim"] = {k: sim.get(k) for k in ("passed", "percent", "engine")}
                summary.write(json.dumps(rec) + "\n")
                summary.flush()
                with lock:
                    print(f"[s{seed}] done after {time.time() - t0:.0f}s: {json.dumps(rec)}", flush=True)
            time.sleep(2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
