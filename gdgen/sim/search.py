"""Finds an input sequence that survives a level, or reports how far the best attempt got.

Breadth-first over ticks: every surviving state branches on hold/release every `input_period` ticks, and states
that are equal after quantization are merged. Because the game is deterministic and x advances the same way
for every survivor, the frontier stays small on real levels.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Hashable, Protocol


class SimState(Protocol):
    x: float
    dead: bool

    def key(self) -> Hashable: ...


@dataclass
class SearchResult:
    passed: bool
    max_x: float
    ticks: int
    inputs: list[bool] = field(default_factory=list)   # hold flag per tick of the best path
    frontier_peak: int = 0


def find_path(
    initial: SimState,
    step: Callable[[SimState, bool], SimState],
    goal_x: float,
    max_ticks: int,
    input_period: int = 1,
    max_frontier: int = 4000,
    key: Callable[[SimState], Hashable] | None = None,
    time_limit: float | None = None,
    bucket: Callable[[SimState], Hashable] | None = None,
) -> SearchResult:
    """time_limit (seconds) ends the search early with the best attempt so far (passed=False).
    bucket groups states (e.g. by mode and height band) so an over-full frontier keeps every group represented."""
    key = key or (lambda st: st.key())
    deadline = time.monotonic() + time_limit if time_limit else None
    frontier: list[tuple[SimState, int]] = [(initial, -1)]   # (state, index into history)
    history: list[tuple[int, bool]] = []                     # (parent index, hold) per kept node
    best_x, best_node, peak = initial.x, -1, 1
    for tick in range(max_ticks):
        branch = tick % input_period == 0
        next_frontier: dict[Hashable, tuple[SimState, int]] = {}
        for state, node in frontier:
            options = (False, True) if branch else (history[node][1] if node >= 0 else False,)
            for hold in options:
                nxt = step(state, hold)
                if nxt.dead:
                    continue
                k = key(nxt)
                if k in next_frontier:
                    continue
                history.append((node, hold))
                next_frontier[k] = (nxt, len(history) - 1)
        if not next_frontier:
            break
        frontier = list(next_frontier.values())
        if len(frontier) > max_frontier:
            frontier = _prune(frontier, max_frontier, key, bucket)
        peak = max(peak, len(frontier))
        lead_state, lead_node = max(frontier, key=lambda e: e[0].x)
        if lead_state.x > best_x:
            best_x, best_node = lead_state.x, lead_node
        if best_x >= goal_x:
            return SearchResult(True, best_x, tick + 1, _inputs(history, best_node), peak)
        if deadline and tick % 64 == 0 and time.monotonic() > deadline:
            break
    inputs = _inputs(history, best_node)
    return SearchResult(False, best_x, len(inputs), inputs, peak)


def _spread(items: list, n: int) -> list:
    """n items evenly spaced over the list."""
    stride = len(items) / n
    return [items[int(i * stride)] for i in range(n)]


def _prune(frontier: list, max_frontier: int, key, bucket) -> list:
    """Keep a spread of states rather than an arbitrary prefix: evenly over the sorted keys, or, with buckets,
    an equal share of every bucket first (small buckets entirely) and the rest spread over the remaining states."""
    frontier.sort(key=lambda e: repr(key(e[0])))
    if bucket is None:
        return _spread(frontier, max_frontier)
    groups: dict[Hashable, list] = {}
    for e in frontier:
        groups.setdefault(bucket(e[0]), []).append(e)
    quota = max(1, max_frontier // len(groups))
    kept, rest = [], []
    for g in groups.values():
        if len(g) <= quota:
            kept += g
        else:
            picked = _spread(g, quota)
            kept += picked
            chosen = {id(e) for e in picked}
            rest += [e for e in g if id(e) not in chosen]
    if len(kept) >= max_frontier:
        return _spread(kept, max_frontier)
    return kept + (_spread(rest, max_frontier - len(kept)) if rest else [])


def _inputs(history: list[tuple[int, bool]], node: int) -> list[bool]:
    out = []
    while node >= 0:
        parent, hold = history[node]
        out.append(hold)
        node = parent
    return out[::-1]
