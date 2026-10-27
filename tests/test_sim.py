"""Simulator checks on hand-built levels whose outcome is known from playing the game."""
import pytest

from gdgen.sim.collide import Simulator
from gdgen.sim.level import build_level
from gdgen.sim.search import find_path

SPIKE, BLOCK, YELLOW_ORB, YELLOW_PAD, SHIP_PORTAL = 8, 1, 36, 35, 13


def obj(object_id: int, col: int, row: int = 0, **extra) -> dict:
    """Object at grid column/row (level-string coordinates: cell centres at 15 + 30*n)."""
    return {"1": str(object_id), "2": str(15 + 30 * col), "3": str(15 + 30 * row), **{k: str(v) for k, v in extra.items()}}


def solve(objects, header=None, end_col=None, input_period=2):
    level = build_level(header or {"kA2": "0", "kA4": "0"}, objects)
    sim = Simulator(level)
    goal = 15 + 30 * (end_col if end_col is not None else max(int(float(o["2"])) // 30 for o in objects) + 3)
    res = find_path(sim.initial_state(), sim.step, goal_x=goal, max_ticks=3000,
                    input_period=input_period, key=sim.key)
    return res, sim


def run_inputs(objects, inputs, header=None):
    level = build_level(header or {"kA2": "0", "kA4": "0"}, objects)
    sim = Simulator(level)
    p = sim.initial_state()
    for hold in inputs:
        p = sim.step(p, hold)
        if p.dead:
            break
    return p


def test_flat_ground_idle_survives():
    p = run_inputs([obj(BLOCK, 200, 5)], [False] * 600)
    assert not p.dead
    assert p.y == pytest.approx(105.0)
    assert p.x == pytest.approx(600 * 0.9 * 5.7700018882751465 * 0.25, rel=1e-6)


def test_spike_kills_without_input():
    p = run_inputs([obj(SPIKE, 10)], [False] * 400)
    assert p.dead


def test_single_and_triple_spike_are_jumpable():
    assert solve([obj(SPIKE, 10)])[0].passed
    assert solve([obj(SPIKE, c) for c in (10, 11, 12)])[0].passed


def test_five_spikes_are_impossible_at_normal_speed():
    assert not solve([obj(SPIKE, c) for c in range(10, 15)])[0].passed


def test_two_block_step_up_is_possible_three_is_not():
    two = [obj(BLOCK, c, r) for c in range(10, 20) for r in (0, 1)]
    assert solve(two)[0].passed
    three = [obj(BLOCK, c, r) for c in range(10, 20) for r in (0, 1, 2)]
    assert not solve(three)[0].passed


def test_landing_on_block_and_running_along_it():
    level = [obj(BLOCK, c, 0) for c in range(10, 30)]
    res, _ = solve(level + [obj(SPIKE, c, 0) for c in range(30, 34)] + [obj(BLOCK, c, 0) for c in range(34, 40)])
    assert res.passed


def test_yellow_orb_clears_a_long_spike_pit():
    spikes = [obj(SPIKE, c) for c in range(10, 17)]   # 7 spikes: too long for a jump, fine with the orb
    assert not solve(spikes)[0].passed
    assert solve(spikes + [obj(YELLOW_ORB, 11, 2)])[0].passed


def test_yellow_pad_launches_without_input():
    sim = Simulator(build_level({"kA2": "0", "kA4": "0"}, [obj(YELLOW_PAD, 10)]))
    p, peak = sim.initial_state(), 0.0
    for _ in range(400):
        p = sim.step(p, False)
        peak = max(peak, p.y)
    assert not p.dead
    # 16 units/frame launch: apex ~ 16^2 / (2 * 0.958) * 0.9 ~ 120 above the floor (cube jump: ~67).
    assert peak - 105 > 100


def test_ship_corridor_flight():
    # Ship from the start; a wall with a 3-block gap in the middle of the corridor.
    wall = [obj(BLOCK, 30, r) for r in range(0, 10) if r not in (4, 5, 6)]
    res, _ = solve(wall, header={"kA2": "1", "kA4": "0"}, end_col=40, input_period=2)
    assert res.passed


def test_bucketed_pruning_keeps_every_group():
    from gdgen.sim.search import _prune
    states = [(type("S", (), {"y": y, "g": g})(), i) for i, (g, y) in enumerate([("ship", y) for y in range(1000)]
                                                                              + [("cube", 0), ("wave", 5)])]
    kept = _prune(list(states), 20, key=lambda s: (s.g, s.y), bucket=lambda s: s.g)
    groups = {s.g for s, _ in kept}
    assert len(kept) == 20 and groups == {"ship", "cube", "wave"}
    plain = _prune(list(states), 20, key=lambda s: (s.g, s.y), bucket=None)
    assert len(plain) == 20
