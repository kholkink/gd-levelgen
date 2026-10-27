"""Compares the game's per-tick player trace (replay/out/<name>.csv) with the simulator's (replay/sim/<name>.csv).

The game logs, at the start of step t, the state left by step t-1; the simulator's row t is the state after t
steps, so rows are compared by tick. The x origin may differ (start position), so x is compared after aligning
on the first tick; y, y velocity, mode and size are compared directly.

Usage: python scripts/compare_trace.py [name ...]     (default: every job with both traces)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT, SIM = ROOT / "replay" / "out", ROOT / "replay" / "sim"
MODE_COLS = ["ship", "ufo", "ball", "wave", "robot", "spider", "swing"]


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def mode(row: dict) -> str:
    return next((m for m in MODE_COLS if row[m] == "1"), "cube")


def compare(name: str, tol_y: float = 0.01, tol_v: float = 0.0015) -> None:
    game, sim = load(OUT / f"{name}.csv"), load(SIM / f"{name}.csv")
    if not game or not sim:
        print(f"{name}: empty trace")
        return
    halves = sum(1 for r in game if r["half"] == "1")
    x0 = float(game[0]["x"]) - float(sim[0]["x"])
    n = min(len(game), len(sim))
    first_bad, worst = None, (0.0, 0.0, 0.0)
    for i in range(n):
        g, s = game[i], sim[i]
        dx = float(g["x"]) - float(s["x"]) - x0
        dy = float(g["y"]) - float(s["y"])
        dv = float(g["y_vel"]) - float(s["y_vel"])
        worst = (max(worst[0], abs(dx)), max(worst[1], abs(dy)), max(worst[2], abs(dv)))
        mismatch = abs(dy) > tol_y or abs(dv) > tol_v or mode(g) != mode(s) or g["upside"] != s["upside"]
        if mismatch and first_bad is None:
            first_bad = i
    status = "MATCH" if first_bad is None else f"DIVERGES at tick {first_bad}"
    print(f"{name:24s} ticks game={len(game):5d} sim={len(sim):5d} half-ticks={halves}  x-offset={x0:+.3f}  "
          f"max|dx|={worst[0]:.4f} max|dy|={worst[1]:.4f} max|dv|={worst[2]:.4f}  {status}")
    if first_bad is not None:
        for i in range(max(0, first_bad - 3), min(n, first_bad + 4)):
            g, s = game[i], sim[i]
            print(f"    t={i:5d}  game y={float(g['y']):9.4f} v={float(g['y_vel']):8.4f} {mode(g):6s} grd={g['on_ground']}"
                  f"   sim y={float(s['y']):9.4f} v={float(s['y_vel']):8.4f} {mode(s):6s} grd={s['on_ground']}")


def main() -> int:
    names = sys.argv[1:] or sorted(p.stem for p in OUT.glob("*.csv") if (SIM / p.name).exists())
    results = OUT / "results.txt"
    if results.exists():
        print(results.read_text())
    for name in names:
        compare(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
