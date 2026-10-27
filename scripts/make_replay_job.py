"""Builds a job for the gdgen replay mod (replay-mod/) and the matching simulator traces.

Jobs are either probe levels generated here (simple layouts with scripted inputs, to compare raw physics) or
downloaded levels with the input sequence our path search finds. For every job the simulator's own trace is
written next to the game's, so scripts/compare_trace.py can diff them tick by tick.

Usage:
  python scripts/make_replay_job.py --probes                     # physics probes only
  python scripts/make_replay_job.py --probes --levels 139263391  # plus real levels (search for inputs)
Then start Geometry Dash (with Geode + the mod); results land in replay/out/ (the mod's GDGEN_REPLAY_DIR).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gdgen.levelstring import decompress, encode_level_string, parse_level, serialize_level  # noqa: E402
from gdgen.sim import fast  # noqa: E402
from gdgen.sim.collide import Simulator  # noqa: E402
from gdgen.sim.level import build_level  # noqa: E402
from gdgen.sim.search import find_path  # noqa: E402

REPLAY_DIR = ROOT / "replay"
DB = ROOT / "data" / "levels.sqlite"
TRACE_HEADER = "tick,half,x,y,y_vel,ship,ufo,ball,wave,robot,spider,swing,upside,size,speed,on_ground,dead"
MODES = ["ship", "ufo", "ball", "wave", "robot", "spider", "swing"]
BLOCK, SPIKE, YELLOW_ORB, YELLOW_PAD = 1, 8, 36, 35
START_POS_ID = "31"


def base_header(db: sqlite3.Connection) -> dict[str, str]:
    """A real level's header (colours etc.) to build probe levels on."""
    blob, = db.execute("SELECT data FROM levels WHERE status = 'downloaded' LIMIT 1").fetchone()
    header, _ = parse_level(decompress(blob))
    return header


def obj(object_id: int, col: float, row: float = 0) -> dict[str, str]:
    return {"1": str(object_id), "2": f"{15 + 30 * col:g}", "3": f"{15 + 30 * row:g}"}


def hold_ranges(ranges: list[tuple[int, int]], total: int) -> list[bool]:
    """Per-tick hold flags from [start, end) tick ranges."""
    holds = [False] * total
    for a, b in ranges:
        for t in range(a, min(b, total)):
            holds[t] = True
    return holds


def probes(header: dict[str, str]) -> list[tuple[str, dict, list[dict], list[bool]]]:
    def with_mode(mode: str, speed: str = "0") -> dict:
        h = dict(header)
        h.update({"kA2": mode, "kA3": "0", "kA4": speed, "kA8": "0", "kA11": "0"})
        return h

    far = [obj(BLOCK, 400, 8)]       # sets the level length; well above the player's path
    out = []
    # Cube: single taps, a long hold (re-jumps on landing), taps at three speeds.
    out.append(("probe_cube_taps", with_mode("0"), far, hold_ranges([(60, 61), (300, 304), (520, 640)], 900)))
    out.append(("probe_cube_slow", with_mode("0", "1"), far, hold_ranges([(60, 61), (300, 420)], 700)))
    out.append(("probe_cube_fast", with_mode("0", "2"), far, hold_ranges([(60, 61), (300, 420)], 700)))
    # Landing on a two-block step and walking off its edge.
    step = [obj(BLOCK, c, r) for c in range(14, 22) for r in (0, 1)]
    out.append(("probe_cube_step", with_mode("0"), step + far, hold_ranges([(84, 85)], 600)))
    # Ship: hold/release pattern touching both corridor floor and ceiling.
    out.append(("probe_ship", with_mode("1"), far, hold_ranges([(20, 200), (260, 300), (330, 520)], 900)))
    # UFO taps, ball flips, wave zig-zag, robot holds.
    out.append(("probe_ufo", with_mode("3"), far, hold_ranges([(30, 31), (90, 91), (150, 151), (155, 156)], 600)))
    out.append(("probe_ball", with_mode("2"), far, hold_ranges([(40, 41), (200, 201)], 600)))
    out.append(("probe_wave", with_mode("4"), far, hold_ranges([(20, 120), (200, 260)], 500)))
    out.append(("probe_robot", with_mode("5"), far, hold_ranges([(40, 41), (200, 280)], 600)))
    # Orb and pad.
    orb = [obj(YELLOW_ORB, 10, 2)]
    out.append(("probe_orb", with_mode("0"), orb + far, hold_ranges([(176, 177), (208, 230)], 600)))
    out.append(("probe_pad", with_mode("0"), [obj(YELLOW_PAD, 10)] + far, [False] * 600))
    return out


def sim_trace(sim: Simulator, holds: list[bool]) -> list[str]:
    """Rows in the mod's format; row t is the state after t steps (what the mod logs at tick t)."""
    rows = []
    p = sim.initial_state()
    for t in range(len(holds) + 1):
        flags = [1 if p.mode == m else 0 for m in MODES]
        rows.append(",".join(str(v) for v in [t, 0, f"{p.x:.4f}", f"{p.y:.4f}", f"{p.y_vel:.4f}", *flags,
                                                1 if p.upside_down else 0, f"{p.vehicle_size:.2f}",
                                                f"{p.speed:.2f}", 1 if p.on_ground else 0, 1 if p.dead else 0]))
        if t == len(holds) or p.dead:
            break
        p = sim.step(p, holds[t])
    return rows


def input_events(holds: list[bool]) -> list[str]:
    events, prev = [], False
    for t, h in enumerate(holds):
        if h != prev:
            events.append(f"{t}:{1 if h else 0}")
            prev = h
    if prev:
        events.append(f"{len(holds)}:0")
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--levels", type=int, nargs="*", default=[])
    ap.add_argument("--time-scale", type=float, default=2.0)
    ap.add_argument("--quit-when-done", action="store_true")
    args = ap.parse_args()

    (REPLAY_DIR / "levels").mkdir(parents=True, exist_ok=True)
    (REPLAY_DIR / "sim").mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    jobs = []

    if args.probes:
        header = base_header(db)
        for name, hdr, objects, holds in probes(header):
            text = serialize_level(hdr, objects)
            (REPLAY_DIR / "levels" / f"{name}.txt").write_text(encode_level_string(text))
            sim = Simulator(build_level(hdr, objects))
            (REPLAY_DIR / "sim" / f"{name}.csv").write_text(TRACE_HEADER + "\n" + "\n".join(sim_trace(sim, holds)) + "\n")
            jobs.append((name, f"levels/{name}.txt", 0, 0, len(holds) + 5, holds))

    for level_id in args.levels:
        row = db.execute("SELECT data, song_id, official_song FROM levels WHERE id = ?", (level_id,)).fetchone()
        if not row:
            print(f"level {level_id} not in the database")
            continue
        blob, song_id, official = row
        header, objects = parse_level(decompress(blob))
        # Levels are loaded as editor levels, where the game honours leftover StartPos objects (ignored online).
        objects = [o for o in objects if o.get("1") != START_POS_ID]
        level = build_level(header, objects)
        sim = Simulator(level)
        ticks = int(level.length / (0.7 * 5.98 * 0.25)) + 240
        if fast.available():             # same search, bit-identical core; the trace below still comes from Python
            res = fast.FastSimulator(level).find_path(goal_x=level.length, max_ticks=ticks, input_period=4,
                                                      max_frontier=300, time_limit=120, bucket=False)
        else:
            res = find_path(sim.initial_state(), sim.step, goal_x=level.length, max_ticks=ticks, input_period=4,
                            max_frontier=300, key=sim.key)
        print(f"level {level_id}: search {'passed' if res.passed else 'failed'} at x={res.max_x:.0f}")
        name = f"level_{level_id}"
        (REPLAY_DIR / "levels" / f"{name}.txt").write_text(encode_level_string(serialize_level(header, objects)))
        (REPLAY_DIR / "sim" / f"{name}.csv").write_text(TRACE_HEADER + "\n" + "\n".join(sim_trace(sim, res.inputs)) + "\n")
        jobs.append((name, f"levels/{name}.txt", song_id or 0, official or 0, len(res.inputs) + 240, res.inputs))

    lines = [f"time_scale {args.time_scale}", f"quit_when_done {1 if args.quit_when_done else 0}"]
    for name, level_file, song, track, max_ticks, holds in jobs:
        lines.append(f"job {name} {level_file} {song} {track} {max_ticks} 1")
        lines.append("inputs " + " ".join(input_events(holds)))
    (REPLAY_DIR / "job.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(jobs)} jobs to {REPLAY_DIR / 'job.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
