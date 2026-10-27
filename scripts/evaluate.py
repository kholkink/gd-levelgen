"""Compares generated levels with real featured levels (heavy: run it on a Kaggle CPU kernel, kernel.py push eval).

    python scripts/evaluate.py --gen smoke='gen/*/level.txt' --gen m1='runs/m1/gen/*/level.txt' \
        --ref-db data/levels.sqlite --ref-n 300 --out eval/run1 [--simulate --sim-seconds 60] [--workers 4]

Writes to --out:
    metrics.jsonl   one line per level: source, name, id, the gdgen/evaluate.py metrics, the generation report
                    (decode errors) and with --simulate two simulations: the whole level (sim) and its gameplay
                    objects alone (sim_layout, what generate.py --select-segments checks)
    summary.json    gdgen.evaluate.compare of every source against the reference ("ref")
    summary.md      the same as a table
    levels/<source>/<name>.txt   the first --copy-levels levels of every source (scripts/gallery.py renders them)
The reference is a seeded random sample of downloaded featured (feature_score > 0), rated, non-platformer levels.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gdgen.evaluate import HISTOGRAMS, SCALARS, compare, level_metrics, simulate  # noqa: E402
from gdgen.objects import is_layout  # noqa: E402
from gdgen.levelstring import decompress, level_text, parse_level  # noqa: E402
from gdgen.sim import fast  # noqa: E402

LIVE_DB = ROOT / "data" / "levels.sqlite"
REF_WHERE = "status = 'downloaded' AND feature_score > 0 AND stars > 0 AND length < 5"
REPORT_KEYS = ("layout_finished", "layout_tokens", "deco_tokens", "layout_decode_errors", "deco_decode_errors")
CFG: dict = {}


def db_uri(path: str) -> str:
    """Read-only; a snapshot (no crawler writing to it) is opened immutable."""
    return f"file:{path}?mode=ro" + ("&immutable=1" if Path(path).resolve() != LIVE_DB.resolve() else "")


def init(cfg: dict) -> None:
    CFG.update(cfg)


def load(kind: str, ref: str) -> tuple[dict, list[dict], str, dict]:
    if kind == "ref":
        db = sqlite3.connect(db_uri(CFG["db"]), uri=True, timeout=60)
        name, blob = db.execute("SELECT name, data FROM levels WHERE id = ?", (int(ref),)).fetchone()
        db.close()
        text = decompress(blob)
        return *parse_level(text), text, {"id": int(ref), "name": name}
    text = level_text(Path(ref).read_text())                   # level.txt or level_string.txt
    report_file = Path(ref).with_name("report.json")
    report = json.loads(report_file.read_text()) if report_file.exists() else {}
    info = {"name": Path(ref).parent.name, "report": {k: report.get(k) for k in REPORT_KEYS},
            "layout_sim": report.get("simulation")}
    return *parse_level(text), text, info


def process(job: tuple[str, str, str, int]) -> dict:
    kind, source, ref, k = job
    t0 = time.time()
    try:
        header, objects, text, info = load(kind, ref)
        m = {"source": source, **info, **level_metrics(header, objects)}
        if CFG["simulate"]:
            for key, objs in (("sim", objects), ("sim_layout", [o for o in objects if is_layout(o)])):
                try:
                    m[key] = simulate(header, objs, CFG["max_frontier"], CFG["sim_seconds"])
                except Exception as e:  # noqa: BLE001  (an unsupported level must not stop the evaluation)
                    m[key] = {"error": f"{type(e).__name__}: {e}"[:300]}
        if k < CFG["copy_levels"]:
            dest = Path(CFG["out"]) / "levels" / source / f"{m['id'] if kind == 'ref' else m['name']}.txt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
            m["file"] = str(dest.relative_to(CFG["out"]))
    except Exception as e:  # noqa: BLE001
        return {"source": source, "ref": ref, "error": f"{type(e).__name__}: {e}"[:300]}
    m["eval_seconds"] = round(time.time() - t0, 1)
    return m


def reference_ids(db: str, n: int, seed: int, where: str) -> list[int]:
    con = sqlite3.connect(db_uri(db), uri=True, timeout=60)
    ids = [r[0] for r in con.execute(f"SELECT id FROM levels WHERE {where} ORDER BY id")]
    con.close()
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))))


def markdown(summary: dict, ref: str) -> str:
    labels = [ref, *(k for k in summary if k != ref)] if ref in summary else list(summary)
    lines = ["| metric | " + " | ".join(f"{lb} (n={summary[lb]['n']})" for lb in labels) + " |",
             "|---|" + "---|" * len(labels)]
    for name in SCALARS:
        cells = []
        for lb in labels:
            q = summary[lb]["metrics"][name]
            cell = f"{q['median']:.3g} [{q['p10']:.3g}..{q['p90']:.3g}]"
            if "in_ref_range" in summary[lb]:
                cell += f" ({100 * summary[lb]['in_ref_range'][name]:.0f}% in ref)"
            cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for h in HISTOGRAMS:
        lines.append(f"| JS {h} | " + " | ".join(f"{summary[lb]['js'][h]:.3f}" if "js" in summary[lb] else "-"
                                                 for lb in labels) + " |")
    for key, title in (("sim", "passable, whole level"), ("sim_layout", "passable, gameplay only")):
        lines.append(f"| {title} | " + " | ".join(
            f"{100 * summary[lb][key]['passed']:.0f}% (median {summary[lb][key]['percent_median']:.0f}%)"
            if key in summary[lb] else "-" for lb in labels) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="append", default=[], help="LABEL=GLOB of generated level.txt files")
    ap.add_argument("--ref-db", help="levels sqlite to sample the reference from")
    ap.add_argument("--ref-n", type=int, default=300)
    ap.add_argument("--ref-where", default=REF_WHERE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--sim-seconds", type=float, default=60)
    ap.add_argument("--max-frontier", type=int, default=256)
    ap.add_argument("--copy-levels", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for spec in args.gen:
        label, _, pattern = spec.partition("=")
        files = sorted(glob.glob(pattern, recursive=True))
        print(f"{label}: {len(files)} generated levels ({pattern})", flush=True)
        jobs += [("gen", label, f, k) for k, f in enumerate(files)]
    if args.ref_db:
        ids = reference_ids(args.ref_db, args.ref_n, args.seed, args.ref_where)
        print(f"ref: {len(ids)} featured levels", flush=True)
        jobs += [("ref", "ref", str(i), k) for k, i in enumerate(ids)]
    if args.simulate:
        print("simulator:", "c++" if fast.available() else "python", flush=True)   # builds the core once, here
    cfg = {"db": args.ref_db, "simulate": args.simulate, "sim_seconds": args.sim_seconds,
           "max_frontier": args.max_frontier, "copy_levels": args.copy_levels, "out": str(out)}

    rows, t0 = [], time.time()
    with open(out / "metrics.jsonl", "w") as f, Pool(args.workers, initializer=init, initargs=(cfg,)) as pool:
        for k, m in enumerate(pool.imap_unordered(process, jobs), 1):
            f.write(json.dumps(m) + "\n")
            f.flush()
            if "error" in m:
                print("error:", m["source"], m["ref"], m["error"], flush=True)
            else:
                rows.append(m)
            if k % 25 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} levels, {time.time() - t0:.0f}s", flush=True)

    sets: dict[str, list[dict]] = {}
    for m in rows:
        sets.setdefault(m["source"], []).append(m)
    summary = compare(sets, "ref")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "summary.md").write_text(markdown(summary, "ref"))
    print(markdown(summary, "ref"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
