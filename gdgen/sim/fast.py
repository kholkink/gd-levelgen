"""The C++ simulator core (gdgen/sim/core/gdsim.cpp) behind the Python Simulator's interface.

    sim = FastSimulator(build_level(header, objects))
    res = sim.find_path(max_frontier=256, time_limit=60)     # the whole search runs in C++
    st = sim.initial_state(); st = sim.step(st, hold)        # single steps; sim.view(st) -> Player fields

Level preparation stays in Python (gdgen/sim/level.py); this module flattens the Level into C records. The
shared library is compiled on first use (g++ / c++ / clang++, -O2 -ffp-contract=off) into gdgen/sim/core/build/
(or $GDGEN_SIM_BUILD), keyed by the source hash, so editing gdsim.cpp rebuilds it. tests/test_fast_sim.py checks
that steps are bit-identical to the Python Simulator.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from .collide import is_flipped
from .level import Level
from .physics import Player
from .search import SearchResult

CORE_DIR = Path(__file__).parent / "core"
SOURCE = CORE_DIR / "gdsim.cpp"
ABI_VERSION = 2

MODES = ["cube", "ship", "ball", "ufo", "wave", "robot", "spider", "swing"]
MODE_INDEX = {m: i for i, m in enumerate(MODES)}
CATEGORIES = {"solid": 1, "slope": 2, "hazard": 3, "speed": 4}
TYPES = {
    **{t: 1 for t in ("CubePortal", "ShipPortal", "BallPortal", "UfoPortal", "WavePortal", "RobotPortal",
                      "SpiderPortal", "SwingPortal")},
    "InverseGravityPortal": 2, "NormalGravityPortal": 3, "GravityTogglePortal": 4,
    "YellowJumpPad": 5, "PinkJumpPad": 6, "RedJumpPad": 7, "GravityPad": 8,
    "YellowJumpRing": 9, "PinkJumpRing": 10, "RedJumpRing": 11, "GravityRing": 12, "GreenRing": 13, "DropRing": 14,
    "CustomRing": 15, "SpiderOrb": 16, "DashRing": 17, "GravityDashRing": 18, "TeleportOrb": 19,
    "TeleportPortal": 20, "SpiderPad": 21, "RegularSizePortal": 22, "MiniSizePortal": 23, "Breakable": 24,
    "Slope": 25, "DualPortal": 26,
}


class ObjRec(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_int32) for n in (
        "index", "cat", "type", "gamemode", "flip_x", "flip_y", "flipped", "multi_activate", "passable", "free_mode",
        "indexed", "has_obb", "t354", "t345", "t347", "t351", "t352", "t353", "t443", "tg_ok", "slope_dir", "pad2")]
        + [(n, ctypes.c_double) for n in ("x", "y", "x0", "y0", "x1", "y1", "radius", "rotation", "speed")]
        + [("obb", ctypes.c_double * 8)]
        + [(n, ctypes.c_double) for n in ("t51", "t54", "t346", "t348", "t349", "t350", "tg_x", "tg_y", "slope_theta")])


class SettingsRec(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_int32) for n in ("start_mode", "start_mini", "start_upside_down", "fix_gravity_bug",
                                               "fix_radius_collision", "pad")]
                + [(n, ctypes.c_double) for n in ("start_speed", "spawn_y", "max_gameplay_y", "length")])


class TrigRec(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_int32) for n in ("kind", "target", "spawn_triggered", "easing", "lock_x", "enable")]
                + [(n, ctypes.c_double) for n in ("x", "dx", "dy", "duration", "rate", "mod_x", "delay")])


class SetRec(ctypes.Structure):
    _fields_ = [("always", ctypes.c_int32), ("pad", ctypes.c_int32), ("lo", ctypes.c_double), ("hi", ctypes.c_double)]


_VIEW_DOUBLES = ("x", "y", "prev_x", "prev_y", "y_vel", "speed", "robot_boost", "gravity_mod", "spider_tp_time",
                 "dash_slope", "time", "last_flip_time", "last_mode_change_time", "cor_min", "cor_max",
                 "corridor_prev_max", "pending_speed", "slope_y_vel", "slope_angle", "slope_vs", "slope_enter_time")
_VIEW_INTS = ("mode", "mini", "upside_down", "held", "buffer", "ring_held_snap", "ring_buf_snap", "used_normal_ring",
              "on_ground", "grounded", "is_rising", "outside_clamp", "jump_released", "spider_pending", "is_dashing",
              "teleported", "fix_gravity_bug", "dead", "killer", "oob_prev", "has_corridor", "free_mode",
              "has_pending_speed", "prev_top_uid", "prev_bottom_uid", "on_slope", "was_on_slope", "slope_ceil",
              "slope_uid", "slope_desc", "slope_clamped")
_VIEW_BOOLS = ("mini", "upside_down", "held", "buffer", "ring_held_snap", "ring_buf_snap", "used_normal_ring",
               "on_ground", "grounded", "is_rising", "outside_clamp", "jump_released", "spider_pending", "is_dashing",
               "teleported", "fix_gravity_bug", "dead", "oob_prev", "free_mode", "on_slope", "was_on_slope", "slope_ceil",
               "slope_desc", "slope_clamped")


class StateView(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_double) for n in _VIEW_DOUBLES] + [("tick", ctypes.c_int64)]
                + [(n, ctypes.c_int32) for n in _VIEW_INTS]
                + [(n, ctypes.c_int32) for n in ("n_touched", "n_activated", "n_speed_log")]
                + [("touched", ctypes.c_int32 * 32), ("activated", ctypes.c_int32 * 128),
                   ("speed_log", ctypes.c_double * (3 * 64))])


class SearchOut(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_int32) for n in ("passed", "ticks", "frontier_peak", "n_inputs")]
                + [("max_x", ctypes.c_double)])


def _compile() -> Path:
    src = SOURCE.read_bytes()
    build = Path(os.environ.get("GDGEN_SIM_BUILD", CORE_DIR / "build"))
    out = build / f"libgdsim-{hashlib.sha1(src).hexdigest()[:12]}.so"
    if out.exists():
        return out
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not cxx:
        raise RuntimeError("no C++ compiler found for the simulator core (set CXX)")
    build.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    subprocess.run([cxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-ffp-contract=off", "-fno-fast-math",
                    "-o", str(tmp), str(SOURCE)], check=True)
    os.replace(tmp, out)
    return out


@lru_cache(maxsize=1)
def library() -> ctypes.CDLL:
    path = os.environ.get("GDGEN_SIM_LIB") or str(_compile())
    lib = ctypes.CDLL(path)
    vp, i32, dbl = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    p32 = ctypes.POINTER(ctypes.c_int32)
    sig = {
        "gd_abi_version": ([], i32), "gd_obj_rec_size": ([], i32), "gd_state_view_size": ([], i32),
        "gd_level_new": ([ctypes.POINTER(ObjRec), i32, ctypes.POINTER(SettingsRec)], vp),
        "gd_level_set_engine": ([vp, ctypes.POINTER(TrigRec), i32, p32, p32, ctypes.POINTER(SetRec), i32, p32, p32,
                                 p32, p32], None),
        "gd_level_free": ([vp], None),
        "gd_initial": ([vp], vp), "gd_step": ([vp, vp, i32], vp), "gd_state_free": ([vp], None),
        "gd_state_view": ([vp, vp, ctypes.POINTER(StateView)], i32),
        "gd_find_path": ([vp, dbl, i32, i32, i32, dbl, i32, ctypes.POINTER(SearchOut),
                          ctypes.POINTER(ctypes.c_uint8), i32], i32),
        "gd_unsupported": ([vp, ctypes.c_char_p, i32], i32),
    }
    for name, (args, res) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res
    if lib.gd_abi_version() != ABI_VERSION:
        raise RuntimeError("simulator core ABI mismatch: rebuild gdsim.cpp")
    if lib.gd_obj_rec_size() != ctypes.sizeof(ObjRec) or lib.gd_state_view_size() != ctypes.sizeof(StateView):
        raise RuntimeError("simulator core record layout differs from fast.py")
    return lib


def available() -> bool:
    """True when the core can be built or loaded here (a C++ compiler or $GDGEN_SIM_LIB)."""
    try:
        library()
        return True
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        return False


def _int_array(values: list[int]):
    return (ctypes.c_int32 * max(1, len(values)))(*values)


def _num(params: dict, key: str, default: float = 0.0) -> float:
    """collide._teleport's num(): float(params.get(k, d) or d)."""
    return float(params.get(key, default) or default)


def _obj_record(o, level: Level) -> ObjRec:
    r = ObjRec()
    r.index, r.cat, r.type = o.index, CATEGORIES.get(o.category, 0), TYPES.get(o.type, 0)
    r.gamemode = MODE_INDEX.get(o.gamemode, 0) if o.gamemode else 0
    r.flip_x, r.flip_y, r.flipped = o.flip_x, o.flip_y, is_flipped(o)
    r.multi_activate, r.passable, r.free_mode = o.multi_activate, o.passable, o.free_mode
    r.indexed = not (o.dynamic and level.engine is not None)
    r.x, r.y, r.x0, r.y0, r.x1, r.y1 = o.x, o.y, o.x0, o.y0, o.x1, o.y1
    r.radius = o.radius or 0.0
    r.rotation = o.rotation
    r.speed = o.speed or 0.0
    r.slope_dir, r.slope_theta = o.slope_dir, o.slope_theta
    if o.obb:
        r.has_obb = 1
        for k, (px, py) in enumerate(o.obb):
            r.obb[2 * k], r.obb[2 * k + 1] = px, py
    if o.type in ("TeleportPortal", "TeleportOrb"):
        params = o.params or {}
        r.t51, r.t54, r.t346 = _num(params, "51"), _num(params, "54"), _num(params, "346")
        r.t348, r.t349 = _num(params, "348"), _num(params, "349")
        r.t350 = _num(params, "350", 1.0) or 1.0
        r.t354 = int(_num(params, "354"))
        r.t345, r.t347 = params.get("345") == "1", params.get("347") == "1"
        r.t351, r.t352, r.t353 = params.get("351") == "1", params.get("352") == "1", params.get("353") == "1"
        r.t443 = params.get("443") == "1"
        if r.t51 > 0:
            group = int(r.t51)
            target = next((q for q in level.objects if group in q.groups), None)
            if target is not None:
                r.tg_ok, r.tg_x, r.tg_y = 1, target.x, target.y
    return r


class FastState:
    """An opaque C++ player state; FastSimulator.view() turns it into Player fields."""

    __slots__ = ("_h", "_lib")

    def __init__(self, handle, lib):
        self._h, self._lib = handle, lib

    def __del__(self):
        if self._h:
            self._lib.gd_state_free(self._h)
            self._h = None


class FastSimulator:
    def __init__(self, level: Level):
        self.level = level
        self._lib = lib = library()
        objs = level.objects
        arr = (ObjRec * max(1, len(objs)))(*[_obj_record(o, level) for o in objs])
        s = SettingsRec(start_mode=MODE_INDEX.get(level.start_mode, 0), start_mini=level.start_mini,
                        start_upside_down=level.start_upside_down, fix_gravity_bug=level.fix_gravity_bug,
                        fix_radius_collision=level.fix_radius_collision, start_speed=level.start_speed,
                        spawn_y=level.spawn_y, max_gameplay_y=level.max_gameplay_y, length=level.length)
        self._h = lib.gd_level_new(arr, len(objs), ctypes.byref(s))
        if level.engine is not None:
            self._set_engine(level.engine)

    def _set_engine(self, engine) -> None:
        trigs, tg, toff = [], [], [0]
        for t in engine.triggers:
            trigs.append(TrigRec(kind=t.kind, target=t.target, spawn_triggered=t.spawn_triggered, easing=t.easing,
                                 lock_x=t.lock_x, enable=t.enable, x=t.x, dx=t.dx, dy=t.dy, duration=t.duration,
                                 rate=t.rate, mod_x=t.mod_x, delay=t.delay))
            tg += list(t.groups)
            toff.append(len(tg))
        always = {id(s) for s in engine._always}
        sets, sg, sgoff, so, sooff = [], [], [0], [], [0]
        for s in engine.sets:
            sets.append(SetRec(always=id(s) in always, lo=s.lo, hi=s.hi))
            sg += list(s.groups)
            sgoff.append(len(sg))
            so += [o.index for o in s.objects]
            sooff.append(len(so))
        trig_arr = (TrigRec * max(1, len(trigs)))(*trigs)
        set_arr = (SetRec * max(1, len(sets)))(*sets)
        self._lib.gd_level_set_engine(self._h, trig_arr, len(trigs), _int_array(tg), _int_array(toff), set_arr,
                                      len(sets), _int_array(sg), _int_array(sgoff), _int_array(so), _int_array(sooff))

    def __del__(self):
        if getattr(self, "_h", None):
            self._lib.gd_level_free(self._h)
            self._h = None

    # ---- the Simulator interface

    def initial_state(self) -> FastState:
        return FastState(self._lib.gd_initial(self._h), self._lib)

    def step(self, state: FastState, hold: bool) -> FastState:
        return FastState(self._lib.gd_step(self._h, state._h, int(bool(hold))), self._lib)

    @property
    def unsupported(self) -> dict[str, float]:
        n = self._lib.gd_unsupported(self._h, None, 0)
        buf = ctypes.create_string_buffer(n + 1)
        self._lib.gd_unsupported(self._h, buf, n + 1)
        out = {}
        for item in buf.value.decode().split(";"):
            if item:
                k, v = item.rsplit("=", 1)
                out[k] = float(v)
        return out

    def find_path(self, goal_x: float | None = None, max_ticks: int | None = None, input_period: int = 1,
                  max_frontier: int = 256, time_limit: float | None = None, bucket: bool = True) -> SearchResult:
        """search.find_path over this level, entirely in C++ (bucket: Simulator.bucket frontier pruning)."""
        goal = self.level.length if goal_x is None else goal_x
        ticks = int(goal / (0.7 * 5.98 * 0.25)) + 240 if max_ticks is None else max_ticks
        out = SearchOut()
        cap = ticks + 1
        inputs = (ctypes.c_uint8 * cap)()
        self._lib.gd_find_path(self._h, goal, ticks, input_period, max_frontier, time_limit or 0.0, int(bucket),
                               ctypes.byref(out), inputs, cap)
        return SearchResult(bool(out.passed), out.max_x, out.ticks, [bool(b) for b in inputs[: out.n_inputs]],
                            out.frontier_peak)

    # ---- inspection

    def view(self, state: FastState) -> dict:
        """The state as Player field values (same names and Python types as gdgen.sim.physics.Player)."""
        v = StateView()
        if self._lib.gd_state_view(self._h, state._h, ctypes.byref(v)):
            raise OverflowError("state lists exceed the view's fixed capacity")
        d = {n: getattr(v, n) for n in _VIEW_DOUBLES + _VIEW_INTS}
        out = {n: d[n] for n in ("x", "y", "prev_x", "prev_y", "y_vel", "speed", "robot_boost", "gravity_mod",
                                 "spider_tp_time", "dash_slope", "time", "last_flip_time", "last_mode_change_time",
                                 "killer", "corridor_prev_max", "prev_top_uid", "prev_bottom_uid", "slope_y_vel",
                                 "slope_angle", "slope_vs", "slope_enter_time", "slope_uid")}
        out.update({n: bool(d[n]) for n in _VIEW_BOOLS})
        out["mode"] = MODES[d["mode"]]
        out["tick"] = v.tick
        out["corridor"] = (d["cor_min"], d["cor_max"]) if d["has_corridor"] else None
        out["pending_speed"] = d["pending_speed"] if d["has_pending_speed"] else None
        out["touched_rings"] = tuple(v.touched[: v.n_touched])
        out["activated"] = frozenset(v.activated[: v.n_activated])
        out["speed_log"] = tuple((int(v.speed_log[3 * i]), v.speed_log[3 * i + 1], v.speed_log[3 * i + 2])
                                 for i in range(v.n_speed_log))
        return out


def player_fields(p: Player) -> dict:
    """A Python Player as the same dict FastSimulator.view() returns."""
    return {name: getattr(p, name) for name in Player.__slots__}
