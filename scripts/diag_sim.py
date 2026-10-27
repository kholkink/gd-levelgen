"""Why does the simulator fail real featured levels (all beatable in the game)? Heavy: Kaggle job `diag`.

For every level: the C++ path search; when it fails, the best path is replayed, and from a state --back ticks before
its end a short exhaustive search (both inputs every tick, deduplicated) records what kills the player near the
furthest x: the killer object (id, category, rotation / scale / hide flags, groups, moved by triggers or not) or the
cause (-2 squished, -3 out of bounds), and the player's mode, height and gravity there.

    python scripts/diag_sim.py --db levels.sqlite --metrics eval/metrics.jsonl --out diag [--gameplay-only]
    python scripts/diag_sim.py --db data/levels.sqlite --ids 96063585 105850197 --out /tmp/diag --workers 1

Writes diag.jsonl (one line per level) and diag_summary.json (killer features over all failed levels).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gdgen.levelstring import decompress, parse_level  # noqa: E402
from gdgen.objects import category_of, is_layout  # noqa: E402
from gdgen.sim import fast  # noqa: E402
from gdgen.sim.level import build_level  # noqa: E402

LIVE_DB = ROOT / "data" / "levels.sqlite"
CAUSES = {-2: "squished", -3: "out of bounds"}
CFG: dict = {}


def db_uri(path: str) -> str:
    return f"file:{path}?mode=ro" + ("&immutable=1" if Path(path).resolve() != LIVE_DB.resolve() else "")


def init(cfg: dict) -> None:
    CFG.update(cfg)


def object_info(level, objects: list[dict], killer: int) -> dict:
    if killer < 0:
        return {"cause": CAUSES.get(killer, str(killer))}
    po = next((o for o in [*level.objects, *level.dynamic_objects] if o.index == killer), None)
    if po is None:
        return {"cause": f"object #{killer} not found"}
    raw = objects[po.source] if 0 <= po.source < len(objects) else {}
    flags = [name for key, name in (("6", "rotated"), ("32", "scaled"), ("128", "scaled"), ("129", "scaled"),
                                    ("135", "hide"), ("136", "non-stick"), ("134", "passable"), ("4", "flip_x"),
                                    ("5", "flip_y")) if raw.get(key) not in (None, "", "0")]
    return {"cause": "object", "id": po.id, "category": category_of(po.id), "x": po.x, "y": po.y,
            "w": round(po.x1 - po.x0, 1), "h": round(po.y1 - po.y0, 1), "flags": sorted(set(flags)),
            "groups": len(po.groups), "dynamic": po.dynamic, "radius": po.radius,
            "keys": sorted(set(raw) - {"1", "2", "3"})[:24]}


def diagnose(header: dict, objects: list[dict]) -> dict:
    if CFG["gameplay_only"]:
        objects = [o for o in objects if is_layout(o)]
    level = build_level(header, objects)
    sim = fast.FastSimulator(level)
    res = sim.find_path(max_frontier=CFG["max_frontier"], time_limit=CFG["seconds"])
    out = {"passed": res.passed, "max_x": res.max_x, "length": level.length,
           "percent": round(100 * min(res.max_x, level.length) / max(level.length, 1.0), 1),
           "start_mode": level.start_mode, "unsupported": dict(sim.unsupported)}
    if res.passed:
        return out
    states = [sim.initial_state()]
    for hold in res.inputs:
        states.append(sim.step(states[-1], hold))
    end = sim.view(states[-1])
    out["end"] = {k: end[k] for k in ("x", "y", "mode", "upside_down", "mini", "speed", "on_ground")}
    frontier, killers, deaths = [states[max(0, len(states) - 1 - CFG["back"])]], Counter(), {}
    for _ in range(CFG["back"] + 240):
        nxt, seen = [], set()
        for st in frontier:
            for hold in (False, True):
                n = sim.step(st, hold)
                v = sim.view(n)
                if v["dead"]:
                    if v["x"] >= res.max_x - 60:
                        killers[v["killer"]] += 1
                        deaths.setdefault(v["killer"], {k: v[k] for k in ("x", "y", "mode", "upside_down", "mini",
                                                                          "y_vel", "on_ground")})
                    continue
                key = (round(v["x"]), round(v["y"]), v["mode"], v["upside_down"], round(v["y_vel"], 1))
                if key not in seen:
                    seen.add(key)
                    nxt.append(n)
        frontier = nxt[:600]
        if not frontier:
            break
    out["killers"] = [{"n": n, **object_info(level, objects, k), "player": deaths[k]}
                      for k, n in killers.most_common(4)]
    return out


def process(level_id: int) -> dict:
    t0 = time.time()
    try:
        db = sqlite3.connect(db_uri(CFG["db"]), uri=True, timeout=60)
        name, blob = db.execute("SELECT name, data FROM levels WHERE id = ?", (level_id,)).fetchone()
        db.close()
        header, objects = parse_level(decompress(blob))
        out = {"id": level_id, "name": name, **diagnose(header, objects)}
    except Exception as e:  # noqa: BLE001
        out = {"id": level_id, "error": f"{type(e).__name__}: {e}"[:300]}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(rows: list[dict]) -> dict:
    failed = [r for r in rows if "killers" in r and r["killers"]]
    top = [r["killers"][0] for r in failed]
    objs = [k for k in top if k["cause"] == "object"]
    pct = lambda n: round(100 * n / max(len(failed), 1), 1)  # noqa: E731
    return {
        "levels": len(rows), "passed": sum(r.get("passed", False) for r in rows), "failed": len(failed),
        "errors": sum("error" in r for r in rows),
        "top_killer_cause": {c: pct(n) for c, n in Counter(k["cause"] for k in top).most_common()},
        "category": {c: pct(n) for c, n in Counter(k["category"] for k in objs).most_common(12)},
        "id": {str(c): pct(n) for c, n in Counter(k["id"] for k in objs).most_common(20)},
        "flags": {c: pct(n) for c, n in Counter(f for k in objs for f in k["flags"]).most_common()},
        "dynamic": pct(sum(k["dynamic"] for k in objs)),
        "no_flags_static": pct(sum(not k["flags"] and not k["dynamic"] for k in objs)),
        "player_mode": {c: pct(n) for c, n in Counter(k["player"]["mode"] for k in top).most_common()},
        "percent_buckets": dict(sorted(Counter(min(9, int(r["percent"] // 10)) * 10 for r in failed).items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--metrics", help="scripts/evaluate.py metrics.jsonl: diagnose its reference levels")
    ap.add_argument("--ids", type=int, nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--gameplay-only", action="store_true", help="drop decoration and triggers first")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--max-frontier", type=int, default=256)
    ap.add_argument("--back", type=int, default=120, help="ticks before the best path's end to search from")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    ids = list(args.ids)
    if args.metrics:
        ids += [r["id"] for r in map(json.loads, Path(args.metrics).read_text().splitlines())
                if r.get("source") == "ref" and "id" in r]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(ids)} levels, gameplay only: {args.gameplay_only}, core: {fast.available()}", flush=True)
    cfg = {"db": args.db, "gameplay_only": args.gameplay_only, "seconds": args.seconds,
           "max_frontier": args.max_frontier, "back": args.back}
    rows = []
    with open(out / "diag.jsonl", "w") as f, Pool(args.workers, initializer=init, initargs=(cfg,)) as pool:
        for k, r in enumerate(pool.imap_unordered(process, ids), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            rows.append(r)
            if k % 25 == 0 or k == len(ids):
                print(f"{k}/{len(ids)}", flush=True)
    summary = summarize(rows)
    (out / "diag_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
