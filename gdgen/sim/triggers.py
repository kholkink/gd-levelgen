"""Trigger engine: toggle, move, stop and spawn triggers evaluated along the player's x trajectory.

In classic levels most triggers fire when the player's x passes the trigger, and the player's x depends only on
time and the speed changes so far (it never depends on button presses). The whole trigger timeline is therefore
a function of the speed history, which states carry as `speed_log`; every search branch with the same speed
history shares one Timeline, and the shifted/enabled objects for a tick are computed once and cached.

Simplifications (counted in TriggerEngine.approximations): touch-triggered triggers fire by x; move triggers
with lock-to-player-Y or move-to-target, and rotate/follow/scale/area/keyframe triggers are not simulated
(objects they drive stay where the other triggers put them).
"""
from __future__ import annotations

import heapq
import math
from bisect import bisect_right
from dataclasses import dataclass, field, replace

from .level import BUCKET, PhysObject

TICKS_PER_SECOND = 240
X_MUL = {0.7: 5.980002, 0.9: 5.7700018882751465, 1.1: 5.870002, 1.3: 6.000002, 1.6: 6.000002}

MOVE, TOGGLE, SPAWN, STOP = 901, 1049, 1268, 1616
UNSUPPORTED_MOVERS = {1346: "rotate", 1347: "follow", 1814: "follow player y", 2067: "scale",
                      3006: "area move", 3007: "area rotate", 3008: "area scale", 3016: "advanced follow",
                      3033: "keyframe"}
MAX_EVENTS = 200_000


def x_step(speed: float) -> float:
    """Horizontal distance per 240 Hz step at a given m_playerSpeed."""
    return speed * X_MUL[speed] * 0.25


def ease(kind: int, rate: float, t: float) -> float:
    """cocos2d easing actions as used by GD's EasingType (0..18)."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    rate = rate if rate > 0 else 2.0
    if kind == 1:
        t *= 2
        return 0.5 * t ** rate if t < 1 else 1.0 - 0.5 * (2.0 - t) ** rate
    if kind == 2:
        return t ** rate
    if kind == 3:
        return t ** (1.0 / rate)
    if kind in (4, 5, 6):
        period = rate
        s = period / 4.0
        if kind == 5:
            t -= 1
            return -(2 ** (10 * t)) * math.sin((t - s) * 2 * math.pi / period)
        if kind == 6:
            return 2 ** (-10 * t) * math.sin((t - s) * 2 * math.pi / period) + 1
        t = t * 2 - 1
        if t < 0:
            return -0.5 * 2 ** (10 * t) * math.sin((t - s) * 2 * math.pi / period)
        return 2 ** (-10 * t) * math.sin((t - s) * 2 * math.pi / period) * 0.5 + 1
    if kind in (7, 8, 9):
        def bounce(u: float) -> float:
            if u < 1 / 2.75:
                return 7.5625 * u * u
            if u < 2 / 2.75:
                u -= 1.5 / 2.75
                return 7.5625 * u * u + 0.75
            if u < 2.5 / 2.75:
                u -= 2.25 / 2.75
                return 7.5625 * u * u + 0.9375
            u -= 2.625 / 2.75
            return 7.5625 * u * u + 0.984375
        if kind == 8:
            return 1 - bounce(1 - t)
        if kind == 9:
            return bounce(t)
        return (1 - bounce(1 - 2 * t)) * 0.5 if t < 0.5 else bounce(2 * t - 1) * 0.5 + 0.5
    if kind == 10:
        t *= 2
        return 0.5 * 2 ** (10 * (t - 1)) if t < 1 else 0.5 * (2 - 2 ** (-10 * (t - 1)))
    if kind == 11:
        return 2 ** (10 * (t - 1)) - 0.001
    if kind == 12:
        return 1 - 2 ** (-10 * t)
    if kind == 13:
        return -0.5 * (math.cos(math.pi * t) - 1)
    if kind == 14:
        return 1 - math.cos(t * math.pi / 2)
    if kind == 15:
        return math.sin(t * math.pi / 2)
    if kind in (16, 17, 18):
        o = 1.70158
        if kind == 17:
            return t * t * ((o + 1) * t - o)
        if kind == 18:
            t -= 1
            return t * t * ((o + 1) * t + o) + 1
        o *= 1.525
        t *= 2
        if t < 1:
            return t * t * ((o + 1) * t - o) / 2
        t -= 2
        return t * t * ((o + 1) * t + o) / 2 + 1
    return t


def _f(obj: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(obj.get(key, default))
    except ValueError:
        return default


def _groups(obj: dict) -> tuple[int, ...]:
    return tuple(int(g) for g in obj.get("57", "").split(".") if g.isdigit())


@dataclass(slots=True)
class Trigger:
    source: int
    kind: int
    x: float
    groups: tuple[int, ...]
    target: int
    spawn_triggered: bool
    touch_triggered: bool
    dx: float = 0.0
    dy: float = 0.0
    duration: float = 0.0
    easing: int = 0
    rate: float = 2.0
    lock_x: bool = False
    mod_x: float = 1.0
    enable: bool = True
    delay: float = 0.0


@dataclass(slots=True)
class MoveSeg:
    start: int
    ticks: int
    dx: float
    dy: float
    easing: int
    rate: float
    lock_x: bool
    mod_x: float
    stop: int = 1 << 60


def parse_triggers(objects: list[dict[str, str]]) -> tuple[list[Trigger], dict[str, int]]:
    triggers, approx = [], {}
    for source, o in enumerate(objects):
        try:
            oid = int(o.get("1", "0"))
        except ValueError:
            continue
        if oid in UNSUPPORTED_MOVERS:
            approx[UNSUPPORTED_MOVERS[oid]] = approx.get(UNSUPPORTED_MOVERS[oid], 0) + 1
            continue
        if oid not in (MOVE, TOGGLE, SPAWN, STOP):
            continue
        t = Trigger(source=source, kind=oid, x=_f(o, "2"), groups=_groups(o), target=int(_f(o, "51")),
                    spawn_triggered=o.get("62") == "1", touch_triggered=o.get("11") == "1")
        if t.touch_triggered:
            approx["touch trigger (fires by x)"] = approx.get("touch trigger (fires by x)", 0) + 1
        if oid == MOVE:
            t.dx, t.dy = _f(o, "28"), _f(o, "29")
            t.duration, t.easing, t.rate = _f(o, "10"), int(_f(o, "30")), _f(o, "85", 2.0)
            t.lock_x = o.get("58") == "1"
            t.mod_x = _f(o, "143", 1.0) or 1.0
            if o.get("59") == "1" or o.get("100") == "1":
                approx["move lock-Y / to-target"] = approx.get("move lock-Y / to-target", 0) + 1
        elif oid == TOGGLE:
            t.enable = o.get("56") == "1"
        elif oid == SPAWN:
            t.delay = _f(o, "63")
        triggers.append(t)
    return triggers, approx


class Timeline:
    """Group enable states and offsets over ticks for one speed history."""

    def __init__(self, engine: "TriggerEngine", speed_log: tuple):
        self.engine = engine
        self.breaks = list(speed_log)                    # (tick, x, x per step)
        self._break_ticks = [b[0] for b in self.breaks]
        self.enabled: dict[int, list[tuple[int, bool]]] = {}
        self.moves: dict[int, list[MoveSeg]] = {}
        self.cache: dict[int, list[PhysObject]] = {}
        self._run()

    def x_at(self, tick: int) -> float:
        i = max(bisect_right(self._break_ticks, tick) - 1, 0)
        t0, x0, v = self.breaks[i]
        return x0 + (tick - t0) * v

    def tick_at(self, x: float) -> int:
        for i, (t0, x0, v) in enumerate(self.breaks):
            last = i + 1 == len(self.breaks)
            if last or x <= self.breaks[i + 1][1]:
                return t0 + max(0, math.ceil((x - x0) / v)) if v > 0 else t0
        return self.breaks[-1][0]

    def group_enabled(self, g: int, tick: int) -> bool:
        changes = self.enabled.get(g)
        if not changes:
            return True
        i = bisect_right(changes, (tick, True)) - 1
        return changes[i][1] if i >= 0 else True

    def group_offset(self, g: int, tick: int) -> tuple[float, float]:
        ox = oy = 0.0
        for seg in self.moves.get(g, ()):
            if tick < seg.start:
                break
            t = min(tick, seg.stop)
            frac = 1.0 if seg.ticks <= 0 else ease(seg.easing, seg.rate, (t - seg.start) / seg.ticks)
            ox += seg.dx * frac
            oy += seg.dy * frac
            if seg.lock_x:
                end = min(t, seg.start + seg.ticks)
                ox += (self.x_at(end) - self.x_at(seg.start)) * seg.mod_x
        return ox, oy

    def _run(self) -> None:
        heap: list[tuple[int, int, Trigger]] = []
        seq = 0
        for trig in self.engine.x_triggers:
            heapq.heappush(heap, (self.tick_at(trig.x), seq, trig))
            seq += 1
        processed = 0
        while heap and processed < MAX_EVENTS:
            tick, _, trig = heapq.heappop(heap)
            processed += 1
            if not all(self.group_enabled(g, tick) for g in trig.groups):
                continue
            if trig.kind == TOGGLE:
                self.enabled.setdefault(trig.target, []).append((tick, trig.enable))
            elif trig.kind == MOVE:
                ticks = round(trig.duration * TICKS_PER_SECOND)
                self.moves.setdefault(trig.target, []).append(
                    MoveSeg(tick + 1, ticks, trig.dx, trig.dy, trig.easing, trig.rate, trig.lock_x, trig.mod_x))
            elif trig.kind == STOP:
                for seg in self.moves.get(trig.target, ()):
                    if seg.start <= tick < seg.start + seg.ticks:
                        seg.stop = tick
            elif trig.kind == SPAWN:
                at = tick + round(trig.delay * TICKS_PER_SECOND)
                for child in self.engine.by_group.get(trig.target, ()):
                    heapq.heappush(heap, (at, seq, child))
                    seq += 1
        if processed >= MAX_EVENTS:
            self.engine.approximations["spawn loop cut"] = 1

    def objects_near(self, tick: int, x0: float, x1: float) -> list[PhysObject]:
        cached = self.cache.get(tick)
        if cached is None:
            cached = self._dynamic_at(tick, x0 - 120.0, x1 + 120.0)
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[tick] = cached
        return [o for o in cached if o.x0 <= x1 and o.x1 >= x0]

    def _dynamic_at(self, tick: int, x0: float, x1: float) -> list[PhysObject]:
        out = []
        for gset in self.engine.sets_near(x0, x1):
            if not all(self.group_enabled(g, tick) for g in gset.groups):
                continue
            dx = dy = 0.0
            for g in gset.groups:
                ox, oy = self.group_offset(g, tick)
                dx += ox
                dy += oy
            for o in gset.objects:
                if o.x1 + dx < x0 or o.x0 + dx > x1:
                    continue
                if dx == 0.0 and dy == 0.0:
                    out.append(o)
                    continue
                out.append(replace(o, x=o.x + dx, y=o.y + dy, x0=o.x0 + dx, y0=o.y0 + dy, x1=o.x1 + dx,
                                   y1=o.y1 + dy,
                                   obb=tuple((px + dx, py + dy) for px, py in o.obb) if o.obb else None))
        return out


@dataclass
class GroupSet:
    """Dynamic objects sharing the same set of trigger-driven groups (they move and toggle together)."""
    groups: tuple[int, ...]
    objects: list[PhysObject]
    lo: float = 0.0
    hi: float = 0.0


@dataclass
class TriggerEngine:
    triggers: list[Trigger]
    dynamic_objects: list[PhysObject]
    driven_groups: set[int]
    approximations: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x_triggers = [t for t in self.triggers if not t.spawn_triggered]
        self.by_group: dict[int, list[Trigger]] = {}
        for t in self.triggers:
            for g in t.groups:
                self.by_group.setdefault(g, []).append(t)
        reach: dict[int, float] = {}
        unbounded: set[int] = set()
        for t in self.triggers:
            if t.kind == MOVE:
                reach[t.target] = reach.get(t.target, 0.0) + abs(t.dx)
                if t.lock_x:
                    unbounded.add(t.target)
        sets: dict[tuple[int, ...], GroupSet] = {}
        for o in self.dynamic_objects:
            key = tuple(sorted(g for g in o.groups if g in self.driven_groups))
            if key not in sets:
                sets[key] = GroupSet(key, [])
            sets[key].objects.append(o)
        self.sets = list(sets.values())
        self._always: list[GroupSet] = []
        self._buckets: dict[int, list[GroupSet]] = {}
        for s in self.sets:
            margin = sum(reach.get(g, 0.0) for g in s.groups)
            s.lo = min(o.x0 for o in s.objects) - margin
            s.hi = max(o.x1 for o in s.objects) + margin
            if any(g in unbounded for g in s.groups) or s.hi - s.lo > 200 * BUCKET:
                self._always.append(s)
                continue
            for b in range(math.floor(s.lo / BUCKET), math.floor(s.hi / BUCKET) + 1):
                self._buckets.setdefault(b, []).append(s)
        self._timelines: dict[tuple, Timeline] = {}

    def sets_near(self, x0: float, x1: float) -> list[GroupSet]:
        seen, out = set(), list(self._always)
        for b in range(math.floor(x0 / BUCKET), math.floor(x1 / BUCKET) + 1):
            for s in self._buckets.get(b, ()):
                if id(s) not in seen:
                    seen.add(id(s))
                    out.append(s)
        return out

    def timeline(self, speed_log: tuple) -> Timeline:
        tl = self._timelines.get(speed_log)
        if tl is None:
            if len(self._timelines) > 64:
                self._timelines.clear()
            tl = self._timelines[speed_log] = Timeline(self, speed_log)
        return tl
