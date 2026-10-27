"""Level metrics for comparing generated levels with real featured ones (scripts/evaluate.py, scripts/gallery.py).

Per level: object counts of the two streams (gameplay / decoration), density per 300-unit segment, empty segments
and the longest gap in the gameplay, vertical spread, hazards, orbs and pads, portals, triggers, groups, color
channels, the duration through the speed portals, object id and category histograms, and optionally passability
through the simulator (the C++ core when it can be built). `compare` reduces sets of per-level metrics to per-metric
quantiles, the share of a set inside the reference's 10-90 % range and Jensen-Shannon distances of the histograms.
"""
from __future__ import annotations

import math
import time
from collections import Counter

import numpy as np

from gdgen.music import SEG_UNITS, LevelClock
from gdgen.objects import category_of, is_layout
from gdgen.sim import fast
from gdgen.sim.collide import Simulator
from gdgen.sim.level import build_level
from gdgen.sim.search import find_path

SCALARS = [
    "objects", "gameplay", "deco", "deco_share", "length", "seconds", "gameplay_segments", "empty_share",
    "gameplay_per_segment", "deco_per_segment", "max_gap", "y_spread", "hazard_share", "interactive_per_segment",
    "portals", "triggers", "groups", "color_channels",
]
HISTOGRAMS = ["categories", "ids_gameplay", "ids_deco"]
SIMULATIONS = ["sim", "sim_layout"]          # whole level / gameplay objects only (what generate.py selects on)
PORTAL_CATEGORIES = {"portal_mode", "portal_gravity", "portal_mirror", "portal_size", "portal_dual", "teleport",
                     "speed"}


def simulate(header: dict, objects: list[dict], max_frontier: int, seconds: float, goal_x: float | None = None) -> dict:
    """Path search over the level (to goal_x instead of its end when given): the C++ core when it can be built here,
    else the Python simulator."""
    level = build_level(header, objects)
    goal = level.length if goal_x is None else min(goal_x, level.length)
    ticks = int(goal / (0.7 * 5.98 * 0.25)) + 240
    t0 = time.time()
    if fast.available():
        sim, engine = fast.FastSimulator(level), "c++"
        res = sim.find_path(goal_x=goal, max_ticks=ticks, max_frontier=max_frontier, time_limit=seconds)
    else:
        sim, engine = Simulator(level), "python"
        res = find_path(sim.initial_state(), sim.step, goal_x=goal, max_ticks=ticks, max_frontier=max_frontier,
                        key=sim.key, time_limit=seconds, bucket=sim.bucket)
    dt = time.time() - t0
    return {"passed": res.passed, "timed_out": not res.passed and dt >= seconds, "max_x": res.max_x,
            "length": level.length, "goal": goal, "seconds": round(dt, 1), "engine": engine,
            "percent": round(100 * min(res.max_x, goal) / max(goal, 1.0), 1),
            "unsupported": dict(sim.unsupported)}


def _num(o: dict, key: str) -> float:
    try:
        return float(o.get(key, 0) or 0)
    except ValueError:
        return 0.0


def _id(o: dict) -> int:
    try:
        return int(o.get("1", "0"))
    except ValueError:
        return 0


def level_metrics(header: dict, objects: list[dict]) -> dict:
    flags = [is_layout(o) for o in objects]
    gameplay = [o for o, g in zip(objects, flags) if g]
    n, ng = len(objects), len(gameplay)
    xs = sorted(_num(o, "2") for o in gameplay)
    ys = np.array([_num(o, "3") for o in gameplay]) if gameplay else np.zeros(1)
    cats = Counter(category_of(_id(o)) for o in objects)
    segs = Counter(int(x // SEG_UNITS) for x in xs)
    span = max(segs) - min(segs) + 1 if segs else 0
    groups = {g for o in objects for g in o.get("57", "").split(".") if g.isdigit()}
    length = xs[-1] if xs else 0.0
    return {
        "objects": n, "gameplay": ng, "deco": n - ng, "deco_share": (n - ng) / max(n, 1),
        "length": length, "seconds": LevelClock(header, gameplay).time(length),
        "gameplay_segments": len(segs), "empty_share": 1 - len(segs) / span if span else 1.0,
        "gameplay_per_segment": ng / max(len(segs), 1), "deco_per_segment": (n - ng) / max(span, 1),
        "max_gap": max((b - a for a, b in zip(xs, xs[1:])), default=0.0),
        "y_spread": float(np.percentile(ys, 95) - np.percentile(ys, 5)),
        "hazard_share": cats["hazard"] / max(ng, 1),
        "interactive_per_segment": (cats["orb"] + cats["pad"]) / max(span, 1),
        "portals": sum(cats[c] for c in PORTAL_CATEGORIES), "triggers": cats["trigger"], "groups": len(groups),
        "color_channels": sum(1 for c in header.get("kS38", "").split("|") if c.strip()),
        "categories": dict(cats),
        "ids_gameplay": dict(Counter(str(_id(o)) for o in gameplay)),
        "ids_deco": dict(Counter(str(_id(o)) for o, g in zip(objects, flags) if not g)),
    }


def _mean_distribution(hists: list[dict]) -> dict[str, float]:
    """Average of the per-level normalized histograms (a huge level does not outweigh the others)."""
    out: Counter = Counter()
    hists = [h for h in hists if h and sum(h.values())]
    for h in hists:
        total = sum(h.values())
        for k, v in h.items():
            out[k] += v / total / len(hists)
    return dict(out)


def js_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon distance (base 2, 0 = identical, 1 = disjoint) of two distributions over string keys."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    a = np.array([p.get(k, 0.0) for k in keys]) + 1e-12
    b = np.array([q.get(k, 0.0) for k in keys]) + 1e-12
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2
    js = 0.5 * np.sum(a * np.log2(a / m)) + 0.5 * np.sum(b * np.log2(b / m))
    return float(math.sqrt(max(js, 0.0)))


def compare(sets: dict[str, list[dict]], ref: str) -> dict:
    """{label: {n, metrics: {name: {median, p10, p90}}, in_ref_range: {name: share}, js: {hist: distance}, sim}}."""
    def quantiles(rows, name):
        v = np.array([r[name] for r in rows], dtype=np.float64)
        return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}

    ref_rows = sets.get(ref, [])
    ref_q = {name: quantiles(ref_rows, name) for name in SCALARS} if ref_rows else {}
    ref_hist = {h: _mean_distribution([r[h] for r in ref_rows]) for h in HISTOGRAMS}
    out = {}
    for label, rows in sets.items():
        if not rows:
            continue
        entry = {"n": len(rows), "metrics": {name: quantiles(rows, name) for name in SCALARS}}
        if ref_q and label != ref:
            entry["in_ref_range"] = {
                name: float(np.mean([ref_q[name]["p10"] <= r[name] <= ref_q[name]["p90"] for r in rows]))
                for name in SCALARS}
            entry["js"] = {h: js_distance(_mean_distribution([r[h] for r in rows]), ref_hist[h]) for h in HISTOGRAMS}
        for key in SIMULATIONS:
            sims = [r[key] for r in rows if isinstance(r.get(key), dict) and "passed" in r[key]]
            if sims:
                entry[key] = {"n": len(sims), "passed": float(np.mean([s["passed"] for s in sims])),
                              "percent_median": float(np.median([s["percent"] for s in sims])),
                              "timed_out": float(np.mean([s["timed_out"] for s in sims])),
                              "unsupported": float(np.mean([bool(s["unsupported"]) for s in sims]))}
        out[label] = entry
    return out
