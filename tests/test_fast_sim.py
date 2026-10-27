"""The C++ simulator core must step exactly like the Python Simulator (the reference).

Random levels mix every mechanic both implement (all game modes, gravity/size/speed portals, pads, orbs, dash,
teleports, spider, breakables, saws, rotated hazards, passable blocks, move/toggle/spawn/stop triggers); random
input sequences drive both simulators, and every field of every state must be equal bit for bit.
"""
import random
from functools import lru_cache

import pytest

from gdgen.objects import load_objects
from gdgen.sim import fast
from gdgen.sim.collide import Simulator
from gdgen.sim.level import COLLIDABLE, build_level
from gdgen.sim.search import find_path

pytestmark = pytest.mark.skipif(not fast.available(), reason="no C++ compiler for the simulator core")

SPECIAL_TYPES = [
    "CubePortal", "ShipPortal", "BallPortal", "UfoPortal", "WavePortal", "RobotPortal", "SpiderPortal", "SwingPortal",
    "InverseGravityPortal", "NormalGravityPortal", "GravityTogglePortal", "YellowJumpPad", "PinkJumpPad", "RedJumpPad",
    "GravityPad", "SpiderPad", "YellowJumpRing", "PinkJumpRing", "RedJumpRing", "GravityRing", "GreenRing", "DropRing",
    "CustomRing", "SpiderOrb", "DashRing", "GravityDashRing", "TeleportOrb", "TeleportPortal", "RegularSizePortal",
    "MiniSizePortal",
]


@lru_cache(maxsize=1)
def ids() -> dict[str, list[int]]:
    table = load_objects()
    out: dict[str, list[int]] = {}
    for oid in sorted(table):
        info = table[oid]
        if info.category not in COLLIDABLE:
            continue
        if info.type in SPECIAL_TYPES or info.type == "Breakable":
            out.setdefault(info.type, []).append(oid)
        if info.category == "solid" and info.type == "Solid":
            out.setdefault("solid", []).append(oid)
        if info.category == "hazard":
            out.setdefault("saw" if info.radius else "hazard", []).append(oid)
        if info.category == "slope":
            out.setdefault("slope", []).append(oid)
    out["speed"] = [200, 201, 202, 203, 1334]
    return out


def random_level(seed: int, cols: int = 160) -> tuple[dict, list[dict]]:
    rng = random.Random(seed)
    table = ids()
    header = {"kA2": str(rng.choice([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])), "kA4": str(rng.randrange(5)),
              "kA3": str(int(rng.random() < 0.2)), "kA11": str(int(rng.random() < 0.15)),
              "kA39": str(int(rng.random() < 0.5)), "kA32": str(int(rng.random() < 0.3))}
    objs: list[dict] = []

    def add(oid: int, col: float, row: float, **keys) -> None:
        o = {"1": str(oid), "2": f"{15 + 30 * col:.3f}", "3": f"{15 + 30 * row:.3f}"}
        o.update({k: str(v) for k, v in keys.items()})
        objs.append(o)

    for _ in range(rng.randrange(30, 90)):
        col, row = rng.uniform(3, cols), rng.choice([0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 8, 10])
        kind = rng.choices(["solid", "hazard", "saw", "special", "speed", "slope", "Breakable"],
                           weights=[30, 20, 4, 30, 3, 8, 3])[0]
        keys = {}
        if kind == "special":
            t = rng.choice(SPECIAL_TYPES)
            oid = rng.choice(table[t])
            if rng.random() < 0.3:
                keys["6"] = rng.choice([90, 180, 270, 45, -30, 15])
            if rng.random() < 0.2:
                keys["4"] = 1
            if rng.random() < 0.2:
                keys["5"] = 1
            if rng.random() < 0.15:
                keys["99"] = 1
            if t.endswith("Portal") and t != "TeleportPortal" and rng.random() < 0.2:
                keys["111"] = 1
            if t == "TeleportPortal":
                keys["54"] = rng.choice([60, 120, -90, 300, 45.5])
            if t in ("TeleportPortal", "TeleportOrb"):
                if t == "TeleportOrb" or rng.random() < 0.3:
                    keys["51"] = 12
                for k in ("351", "352", "353", "345", "347", "443"):
                    if rng.random() < 0.25:
                        keys[k] = 1
                if rng.random() < 0.5:
                    keys["346"] = rng.choice([0, 5, -8, 12.5])
                if rng.random() < 0.3:
                    keys["348"], keys["349"], keys["350"] = rng.choice([0, 3]), rng.choice([0, 9]), rng.choice([0, 1.5])
                if rng.random() < 0.3:
                    keys["354"] = rng.randrange(4)
        else:
            oid = rng.choice(table[kind])
            if kind in ("hazard", "saw") and rng.random() < 0.4:
                keys["6"] = rng.choice([90, 180, 270, 45, 30, -60])
            if kind == "solid" and rng.random() < 0.15:
                keys["134"] = 1
            if rng.random() < 0.15:
                keys["32"] = rng.choice([0.5, 2, 1.5])
            if rng.random() < 0.1:
                keys["128"], keys["129"] = rng.choice([0.5, 2]), rng.choice([1, 3])
        if rng.random() < 0.25:
            keys["57"] = rng.choice(["7", "8", "7.8", "12", "9"])
        add(oid, col, row, **keys)

    # ramps: rows of neighbouring slopes (rising, falling, flipped into ceiling slopes)
    for _ in range(rng.randrange(1, 4)):
        start, row = rng.uniform(5, cols - 10), rng.choice([0, 0, 1, 2])
        flip = {"5": 1} if rng.random() < 0.3 else {}
        mirror = {"4": 1} if rng.random() < 0.5 else {}
        for k in range(rng.randrange(2, 7)):
            add(rng.choice(table["slope"]), start + k, row + (k if not mirror else -k) * 0.5, **flip, **mirror)

    # a run of blocks in the groups the triggers drive, and the triggers themselves
    for c in range(rng.randrange(5, 30)):
        add(table["solid"][0], 20 + c, rng.choice([1, 2, 3]), **{"57": rng.choice(["7", "8", "9"])})
    for _ in range(rng.randrange(0, 6)):
        add(901, rng.uniform(0, cols), 0, **{"51": rng.choice([7, 8, 9]), "28": rng.choice([0, 30, -60, 15.5]),
                                             "29": rng.choice([0, 30, -45, 90]), "10": rng.choice([0, 0.5, 1, 2.25]),
                                             "30": rng.randrange(19), "85": rng.choice([2, 1.5, 3]),
                                             "58": int(rng.random() < 0.2)})
    for _ in range(rng.randrange(0, 3)):
        add(1049, rng.uniform(0, cols), 0, **{"51": rng.choice([7, 8]), "56": int(rng.random() < 0.5)})
    if rng.random() < 0.5:
        add(1268, rng.uniform(0, cols / 2), 0, **{"51": 30, "63": rng.choice([0, 0.25, 1])})
        add(901, 0, 0, **{"51": 9, "28": 60, "29": 0, "10": 0.5, "62": 1, "57": 30})
    if rng.random() < 0.3:
        add(1616, rng.uniform(0, cols), 0, **{"51": 7})
    return header, objs


def random_inputs(seed: int, n: int) -> list[bool]:
    rng = random.Random(seed * 7919 + 1)
    out, hold = [], False
    while len(out) < n:
        out += [hold] * rng.choice([1, 1, 2, 3, 5, 8, 13, 20, 40])
        hold = not hold
    return out[:n]


def first_difference(a: dict, b: dict) -> str | None:
    for k in a:
        if a[k] != b[k]:
            return f"{k}: python {a[k]!r} vs c++ {b[k]!r}"
    return None


@pytest.mark.parametrize("seed", range(60))
def test_steps_match_python(seed):
    header, objs = random_level(seed)
    level = build_level(header, objs)
    py, cc = Simulator(level), fast.FastSimulator(level)
    p, c = py.initial_state(), cc.initial_state()
    assert first_difference(fast.player_fields(p), cc.view(c)) is None
    for tick, hold in enumerate(random_inputs(seed, 3000)):
        p, c = py.step(p, hold), cc.step(c, hold)
        diff = first_difference(fast.player_fields(p), cc.view(c))
        assert diff is None, f"seed {seed} tick {tick + 1}: {diff}"
        if p.dead:
            break
    assert py.unsupported == cc.unsupported


def test_the_random_levels_exercise_the_mechanics():
    modes, flips, teleports, dashes, deaths, sloped = set(), 0, 0, 0, 0, 0
    for seed in range(60):
        header, objs = random_level(seed)
        sim = Simulator(build_level(header, objs))
        p = sim.initial_state()
        for hold in random_inputs(seed, 3000):
            q = sim.step(p, hold)
            modes.add(q.mode)
            flips += q.upside_down != p.upside_down
            teleports += q.teleported
            dashes += q.is_dashing and not p.is_dashing
            sloped += q.on_slope
            p = q
            if p.dead:
                deaths += 1
                break
    assert len(modes) >= 7 and flips >= 20 and teleports >= 3 and dashes >= 2 and deaths >= 20 and sloped >= 50


@pytest.mark.parametrize("seed", range(40))
def test_searched_paths_step_alike(seed):
    """Inputs from a search survive long, so both simulators walk through far more of the level than random play."""
    header, objs = random_level(500 + seed)
    level = build_level(header, objs)
    py, cc = Simulator(level), fast.FastSimulator(level)
    res = cc.find_path(max_frontier=64, time_limit=20)
    p, c = py.initial_state(), cc.initial_state()
    for tick, hold in enumerate(res.inputs):
        p, c = py.step(p, hold), cc.step(c, hold)
        diff = first_difference(fast.player_fields(p), cc.view(c))
        assert diff is None, f"seed {500 + seed} tick {tick + 1}: {diff}"
    assert not p.dead and p.x == res.max_x


def test_search_matches_python_without_pruning():
    """Unpruned searches agree exactly (pruning orders states differently in the two implementations)."""
    compared = 0
    for seed in range(16):
        header, objs = random_level(1000 + seed, cols=40)
        header["kA2"] = "0"
        level = build_level(header, objs)
        goal, ticks, cap = min(level.length, 1200.0), 150, 500
        sim = Simulator(level)
        ref = find_path(sim.initial_state(), sim.step, goal_x=goal, max_ticks=ticks, max_frontier=cap, key=sim.key)
        if ref.frontier_peak >= cap:
            continue
        got = fast.FastSimulator(level).find_path(goal_x=goal, max_ticks=ticks, max_frontier=cap)
        assert (got.passed, got.max_x, got.ticks, got.inputs) == (ref.passed, ref.max_x, ref.ticks, ref.inputs), seed
        compared += 1
    assert compared >= 8


def test_search_with_buckets_finds_a_real_path():
    """A pruned C++ search returns inputs that the Python simulator replays to the same result."""
    for seed in range(20):
        header, objs = random_level(2000 + seed, cols=60)
        level = build_level(header, objs)
        res = fast.FastSimulator(level).find_path(max_frontier=64)
        sim = Simulator(level)
        p = sim.initial_state()
        for hold in res.inputs:
            p = sim.step(p, hold)
        assert not p.dead and p.x == res.max_x
