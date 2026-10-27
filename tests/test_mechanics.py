"""Mechanics added from spec §15-§22: no touch, passable, rotation snapping, teleport portals, dash orbs."""
import pytest

from gdgen.sim.collide import Simulator
from gdgen.sim.level import build_level

SPIKE, BLOCK, TELEPORT, DASH_ORB = 8, 1, 747, 1704


def obj(object_id: int, col: float, row: float = 0, **keys) -> dict:
    o = {"1": str(object_id), "2": str(15 + 30 * col), "3": str(15 + 30 * row)}
    o.update({k.lstrip("k"): str(v) for k, v in keys.items()})
    return o


def run(objects, holds, header=None):
    sim = Simulator(build_level(header or {"kA2": "0", "kA4": "0"}, objects))
    p, trace = sim.initial_state(), []
    for hold in holds:
        p = sim.step(p, hold)
        trace.append(p)
        if p.dead:
            break
    return p, trace


def test_no_touch_spike_does_not_kill():
    p, _ = run([obj(SPIKE, 10, k121=1)], [False] * 400)
    assert not p.dead


def test_solid_rotation_is_snapped_to_zero():
    level = build_level({"kA2": "0"}, [obj(BLOCK, 10, 2, k6=45)])
    block = level.objects[0]
    assert block.rotation == 0.0 and block.obb is None
    assert block.x1 - block.x0 == pytest.approx(30.0)


def test_passable_block_overhead_is_jumped_through():
    # A block row right above the jump arc kills a cube's head unless it is passable.
    ceiling = [obj(BLOCK, c, 2) for c in range(8, 14)]
    holds = [False] * 230 + [True] + [False] * 200
    solid, _ = run(ceiling, holds)
    passable, _ = run([dict(o, **{"134": "1"}) for o in ceiling], holds)
    assert solid.dead
    assert not passable.dead


def test_passable_block_can_be_landed_on():
    platform = [obj(BLOCK, c, 1, k134=1) for c in range(10, 20)]
    holds = [False] * 220 + [True] + [False] * 150
    p, trace = run(platform, holds)
    assert not p.dead
    assert any(abs(s.y - (90 + 60 + 15)) < 1e-6 and s.on_ground for s in trace)   # standing on the platform top


def test_teleport_portal_moves_player_by_key_54():
    p, trace = run([obj(TELEPORT, 10, 1, k54=300)], [False] * 300)
    jump = next(i for i in range(1, len(trace)) if trace[i].y - trace[i - 1].y > 100)
    assert trace[jump].y == pytest.approx(15 + 30 * 1 + 90 + 300, abs=0.5)
    assert trace[jump].x - trace[jump - 1].x == pytest.approx(trace[2].x - trace[1].x, abs=1e-6)  # x unchanged


def test_45_degree_dash_climbs_one_unit_per_unit_while_held():
    # Rotation -45 points the arrow up-right (spec §18.3): dy = dx on every dashing step.
    holds = [False] * 30 + [True] * 60 + [False] * 60
    p, trace = run([obj(DASH_ORB, 1.5, 0.2, k6=-45)], holds)
    dashing = [i for i, s in enumerate(trace) if s.is_dashing]
    assert len(dashing) > 30, "the dash orb should have fired"
    for i in dashing[1:]:
        if trace[i - 1].is_dashing:
            assert trace[i].y - trace[i - 1].y == pytest.approx(trace[i].x - trace[i - 1].x, abs=1e-6)
    assert not trace[-1].is_dashing                       # released -> dash over
