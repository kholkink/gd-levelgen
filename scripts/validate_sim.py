"""Runs the simulator's path search on downloaded featured levels (all known to be beatable).

A level the search cannot finish points at a simulator gap (missing mechanic, wrong rule) or at gameplay that
depends on triggers we do not simulate. Prints per-level progress and a summary.

Usage: python scripts/validate_sim.py [--limit 50] [--level-id 123] [--input-period 2]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gdgen.levelstring import decompress, parse_level  # noqa: E402
from gdgen.sim.collide import Simulator  # noqa: E402
from gdgen.sim.level import build_level  # noqa: E402
from gdgen.sim.search import find_path  # noqa: E402

DB = Path(__file__).resolve().parent.parent / "data" / "levels.sqlite"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--level-id", type=int)
    ap.add_argument("--input-period", type=int, default=2, help="ticks between input decisions (240 Hz ticks)")
    ap.add_argument("--max-frontier", type=int, default=3000)
    ap.add_argument("--keep-dynamic", action="store_true", help="keep objects moved/toggled by triggers")
    args = ap.parse_args()

    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    if args.level_id:
        rows = db.execute("SELECT id, name, data FROM levels WHERE id = ?", (args.level_id,)).fetchall()
    else:
        rows = db.execute(
            "SELECT id, name, data FROM levels WHERE status = 'downloaded' ORDER BY feature_score LIMIT ?",
            (args.limit,),
        ).fetchall()

    results = []
    for level_id, name, blob in rows:
        header, objects = parse_level(decompress(blob))
        level = build_level(header, objects, skip_dynamic=not args.keep_dynamic)
        sim = Simulator(level)
        start = sim.initial_state()
        ticks = int(level.length / (0.7 * 5.98 * 0.25)) + 240     # enough even at the slowest speed
        t0 = time.time()
        res = find_path(start, sim.step, goal_x=level.length, max_ticks=ticks,
                        input_period=args.input_period, max_frontier=args.max_frontier, key=sim.key)
        dt = time.time() - t0
        pct = 100.0 * min(res.max_x, level.length) / max(level.length, 1.0)
        results.append((level_id, name, res.passed, pct, dt))
        print(f"{level_id:>10} {name[:28]:28s} {level.start_mode:6s} {'PASS' if res.passed else 'fail':4s} "
              f"{pct:5.1f}%  x={res.max_x:8.0f}/{level.length:8.0f}  dyn-skipped {level.dynamic_skipped:5d}  {dt:6.1f}s",
              flush=True)
        if sim.unsupported:
            print("           unsupported:", ", ".join(f"{k}@x={v:.0f}" for k, v in sorted(sim.unsupported.items(), key=lambda kv: kv[1])))

    if results:
        passed = sum(1 for r in results if r[2])
        mean_pct = sum(r[3] for r in results) / len(results)
        print(f"\n{passed}/{len(results)} passed, mean progress {mean_pct:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
