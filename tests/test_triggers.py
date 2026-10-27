"""Trigger engine on hand-built levels: a wall that is impossible to pass unless a trigger removes or moves it."""
import pytest

from gdgen.sim.collide import Simulator
from gdgen.sim.level import build_level
from gdgen.sim.search import find_path
from gdgen.sim.triggers import ease

BLOCK, MOVE, TOGGLE, SPAWN = 1, 901, 1049, 1268
WALL_GROUP, SPAWN_GROUP = 5, 7


def obj(object_id: int, col: float, row: float = 0, **keys) -> dict:
    o = {"1": str(object_id), "2": str(15 + 30 * col), "3": str(15 + 30 * row)}
    o.update({k.lstrip("k"): str(v) for k, v in keys.items()})
    return o


def wall(col: int = 20) -> list[dict]:
    # 4 blocks high: no cube jump clears it.
    return [obj(BLOCK, col, r, k57=WALL_GROUP) for r in range(4)]


def passable(objects) -> bool:
    level = build_level({"kA2": "0", "kA4": "0"}, objects)
    sim = Simulator(level)
    res = find_path(sim.initial_state(), sim.step, goal_x=15 + 30 * 30, max_ticks=3000, input_period=4,
                    key=sim.key)
    return res.passed


def test_wall_alone_is_impassable():
    assert not passable(wall())


def test_toggle_off_before_the_wall_opens_it():
    assert passable(wall() + [obj(TOGGLE, 10, 5, k51=WALL_GROUP, k56=0)])


def test_toggle_after_the_wall_is_too_late():
    assert not passable(wall() + [obj(TOGGLE, 25, 5, k51=WALL_GROUP, k56=0)])


def test_move_trigger_lifts_the_wall():
    lift = obj(MOVE, 10, 5, k51=WALL_GROUP, k28=0, k29=300, k10=0.5, k30=0)
    assert passable(wall() + [lift])


def test_slow_move_does_not_finish_in_time():
    # Starts lifting just before the wall and only 30 units per second: the wall is still there.
    lift = obj(MOVE, 18, 5, k51=WALL_GROUP, k28=0, k29=30, k10=1.0, k30=0)
    assert not passable(wall() + [lift])


def test_spawn_chain_with_delay_toggles_the_wall():
    spawn = obj(SPAWN, 2, 5, k51=SPAWN_GROUP, k63=1.0)
    toggle = obj(TOGGLE, 2, 6, k51=WALL_GROUP, k56=0, k57=SPAWN_GROUP, k62=1)
    assert passable(wall() + [spawn, toggle])


def test_spawn_triggered_toggle_does_not_fire_by_x():
    toggle = obj(TOGGLE, 2, 6, k51=WALL_GROUP, k56=0, k57=SPAWN_GROUP, k62=1)
    assert not passable(wall() + [toggle])


@pytest.mark.parametrize("kind", range(19))
def test_easing_endpoints(kind):
    assert ease(kind, 2.0, 0.0) == 0.0
    assert ease(kind, 2.0, 1.0) == 1.0
