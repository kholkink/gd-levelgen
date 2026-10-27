"""Fetches generated levels into the game as soon as they exist (local and light: one Kaggle listing per poll).

Sources:
    --dataset gdgen-runs-m1-gen   a Colab run artifact with ready-s<seed>-gameplay.txt / ready-s<seed>.txt
                                  (scripts/gen_many.py; the worker publishes the run every --save-every minutes)
    --kernel gd-levelgen-gen      a Kaggle gen kernel: once it completes, every gen/s<seed>/level_string.txt
Every new level goes to replay/import/<prefix> s<seed>[ gameplay].txt; the gdgen replay mod (in a running Geometry
Dash) turns it into a level in Create -> My Levels within two seconds. Already fetched files are remembered in
replay/import/.seen.json.

    python scripts/watch_levels.py --dataset gdgen-runs-m1-gen --prefix m1 [--kernel gd-levelgen-gen] [--every 60]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle"))
from kutil import kaggle_bin, username  # noqa: E402

IMPORT = ROOT / "replay" / "import"
READY = re.compile(r"^ready-s(\d+)(-gameplay)?\.txt$")


def kaggle(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run([kaggle_bin(), *args], capture_output=True, text=True, timeout=timeout)


def deliver(src: Path, name: str) -> None:
    """Copies a level string into the import dir under its level name (whole file first, then a rename)."""
    IMPORT.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    if data[:2] == b"PK":                                   # Kaggle zips some single-file downloads
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            data = z.read(z.namelist()[0])
    tmp = IMPORT / f"{name}.part"
    tmp.write_bytes(data.strip())
    tmp.replace(IMPORT / f"{name}.txt")
    print(time.strftime("%H:%M:%S"), f"-> {name} ({len(data) // 1024} KB)", flush=True)


def poll_dataset(ds: str, prefix: str, seen: dict) -> None:
    r = kaggle("datasets", "files", ds, "--csv", "--page-size", "200")
    if r.returncode != 0:
        return                                              # not published yet
    for row in csv.DictReader(io.StringIO(r.stdout)):
        name, stamp = row.get("name", ""), row.get("creationDate", "")
        m = READY.match(name)
        key = f"{ds}/{name}"
        if not m or seen.get(key) == stamp:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            d = kaggle("datasets", "download", ds, "-f", name, "-p", tmp, "-o", "-q")
            files = [p for p in Path(tmp).iterdir() if p.is_file()]
            if d.returncode != 0 or not files:
                print(f"download of {name} failed: {(d.stdout + d.stderr).strip()[-200:]}", flush=True)
                continue
            deliver(files[0], f"{prefix} s{m.group(1)}{' gameplay' if m.group(2) else ''}")
        seen[key] = stamp


def poll_kernel(kernel: str, prefix: str, seen: dict) -> None:
    ref = f"{username()}/{kernel}"
    status = kaggle("kernels", "status", ref).stdout
    if "COMPLETE" not in status or seen.get(ref):
        return
    with tempfile.TemporaryDirectory() as tmp:
        kaggle("kernels", "output", ref, "-p", tmp, "-o", "-q", "--file-pattern", r"gen/.*/level_string\.txt$",
               timeout=1800)
        for f in sorted(Path(tmp).glob("**/gen/*/level_string.txt")):
            deliver(f, f"{prefix} {f.parent.name}")
    seen[ref] = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="Colab run artifact (e.g. gdgen-runs-m1-gen)")
    ap.add_argument("--prefix", default="gen", help="level name prefix for the dataset's levels")
    ap.add_argument("--kernel", help="Kaggle gen kernel (e.g. gd-levelgen-gen)")
    ap.add_argument("--kernel-prefix", default="kaggle")
    ap.add_argument("--every", type=float, default=60, help="seconds between polls")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    seen_file = IMPORT / ".seen.json"
    seen = json.loads(seen_file.read_text()) if seen_file.exists() else {}
    ds = f"{username()}/{args.dataset}" if args.dataset and "/" not in args.dataset else args.dataset
    print(f"watching {ds or '-'} and {args.kernel or '-'} -> {IMPORT}", flush=True)
    while True:
        try:
            if ds:
                poll_dataset(ds, args.prefix, seen)
            if args.kernel:
                poll_kernel(args.kernel, args.kernel_prefix, seen)
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"poll failed: {e}", flush=True)
        IMPORT.mkdir(parents=True, exist_ok=True)
        seen_file.write_text(json.dumps(seen, indent=1))
        if args.once:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
