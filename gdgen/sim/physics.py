"""Player state and motion for GD 2.2081 (Windows), transcribed from the game's code.

See re/physics_spec.md for the derivation (sections referenced below as §n). Coordinates are in-game units:
one block is 30, the level floor's top surface is y = 90. One physics step is 1/240 s with dt60 = 0.25.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field, fields

STEP_SECONDS = 1.0 / 240.0
DT60 = 0.25                    # GJBaseGameLayer::update: 4 steps per 60-fps frame (0x140237850)
DTJ = 0.225                    # PlayerObject::update -> updateJump(dt60 * 0.9) (0x140388d80)
FLOOR_Y = 90.0

# PlayerObject::updateTimeMod (0x1403a0d50): m_playerSpeed -> (jump accel, gravity, x multiplier).
SPEED_CONSTANTS = {
    0.7: (10.620032, 0.940199, 5.980002),
    0.9: (11.180031776428223, 0.9581990242004395, 5.7700018882751465),
    1.1: (11.420032, 0.957199, 5.870002),
    1.3: (11.230032, 0.961199, 6.000002),
    1.6: (11.230032, 0.961199, 6.000002),
}
VEHICLE_GRAVITY = 0.958199     # §5.0: ball/ship/ufo/wave/swing/spider use a fixed gravity
GROUND_GRAVITY_MULT = {"cube": 1.0, "ball": 0.6, "spider": 0.6, "robot": 0.9}
FLY_MODES = frozenset({"ship", "ufo", "wave", "swing"})
FIXED_GRAVITY_MODES = frozenset({"ship", "ufo", "wave", "swing", "ball", "spider"})
BOUNDED_MODES = frozenset({"ship", "ufo", "wave", "swing", "ball", "spider"})
CORRIDOR_HEIGHT = {"ship": 300.0, "ufo": 300.0, "wave": 300.0, "swing": 300.0, "ball": 240.0, "spider": 270.0}
HITBOX_SIZE = {"wave": 10.0, "spider": 27.0}            # m_width = m_height; 30 otherwise (§2.1)
BOUNDS_HEIGHT = {"wave": 20.0, "spider": 27.0}          # p+2736; 30 otherwise (§2.3)


def _f32(v: float) -> float:
    return struct.unpack("f", struct.pack("f", v))[0]


def set_y_vel(v: float) -> float:
    """PlayerObject::setYVelocity (0x140388d10): keeps three decimals (C round, half away from zero)."""
    t = float(int(v))
    if v == t:
        return v
    frac = (v - t) * 1000.0
    rounded = float(int(frac + (0.5 if frac >= 0 else -0.5)))
    return rounded / 1000.0 + t


@dataclass(slots=True)
class Player:
    x: float = 0.0
    y: float = 105.0
    prev_x: float = 0.0
    prev_y: float = 105.0
    y_vel: float = 0.0
    mode: str = "cube"
    mini: bool = False
    upside_down: bool = False
    speed: float = 0.9
    held: bool = False              # +2437
    buffer: bool = False            # +2438
    ring_held_snap: bool = False    # +2441
    ring_buf_snap: bool = False     # +2442
    used_normal_ring: bool = False  # +2443
    on_ground: bool = True          # +2497
    grounded: bool = True           # +2572
    is_rising: bool = False         # +2588
    outside_clamp: bool = False     # +2386
    jump_released: bool = False     # +2460
    robot_boost: float = 0.0        # +2096
    gravity_mod: float = 1.0        # +2948
    spider_pending: bool = False    # spider jump requested; resolved by move()'s hook (spec §16)
    spider_tp_time: float = -1.0    # p+2080: time of the last spider teleport (0.04 s death grace)
    is_dashing: bool = False        # +2532 (spec §18)
    dash_slope: float = 0.0         # +1400: dy per unit dx while dashing
    teleported: bool = False        # +1376: skip the corridor clamp on the next collision pass (spec §17)
    fix_gravity_bug: bool = False   # +1377: level setting kA32
    time: float = 0.0
    tick: int = 0                   # physics steps completed
    speed_log: tuple = ()           # ((tick, x, x per step), ...) at every speed change; drives the trigger timeline
    last_flip_time: float = 0.0
    last_mode_change_time: float = 0.0
    dead: bool = False
    killer: int = -1                # object index that killed the player; -2 squish, -3 out of bounds
    # Collision bookkeeping that survives between steps.
    oob_prev: bool = False
    corridor: tuple[float, float] | None = None     # (minY, maxY) of the bounded-mode corridor
    free_mode: bool = False                         # layer+785: set by the last mode portal (key 111)
    corridor_prev_max: float = 0.0
    pending_speed: float | None = None
    touched_rings: tuple[int, ...] = ()             # rings overlapped during the last collision pass
    activated: frozenset = field(default_factory=frozenset)   # single-use objects already used
    prev_top_uid: int = -1
    prev_bottom_uid: int = -1
    # slopes (spec §21)
    on_slope: bool = False          # +2480
    was_on_slope: bool = False      # +2481
    slope_y_vel: float = 0.0        # +2484: launch velocity when leaving the slope
    slope_angle: float = 0.0        # +2352: signed slope angle, feeds the probe's cos()
    slope_vs: float = 0.0           # +2360: the slope's vertical speed at full x speed
    slope_ceil: bool = False        # +2387: the current slope is a ceiling slope
    slope_uid: int = -1             # +1672: the slope the player is on
    slope_desc: bool = False        # +1676: descending
    slope_enter_time: float = 0.0   # +1432
    slope_clamped: bool = False     # +2384/2385: the last snap was clamped at the slope's far end (the crest)

    @property
    def s(self) -> int:
        return -1 if self.upside_down else 1

    @property
    def vehicle_size(self) -> float:
        return 0.6 if self.mini else 1.0

    @property
    def width(self) -> float:
        return HITBOX_SIZE.get(self.mode, 30.0) * self.vehicle_size

    @property
    def bounds_height(self) -> float:
        return BOUNDS_HEIGHT.get(self.mode, 30.0)

    def constants(self) -> tuple[float, float, float]:
        return SPEED_CONSTANTS[self.speed]


_PLAYER_SLOTS = tuple(f.name for f in fields(Player))
_new_player = object.__new__


def _compile_copy():
    """Player.copy as straight-line attribute copies: the search copies a state on every step."""
    lines = "\n".join(f"    n.{name} = s.{name}" for name in _PLAYER_SLOTS)
    ns = {"_new": _new_player, "Player": Player}
    exec(f"def copy(s):\n    n = _new(Player)\n{lines}\n    return n\n", ns)
    return ns["copy"]


Player.copy = _compile_copy()


def is_falling(p: Player) -> bool:
    """playerIsFallingBugged (0x14039a430), including the upside-down comparison against +2G."""
    g = p.constants()[1]
    if p.mode == "swing" or p.fix_gravity_bug:
        return p.y_vel > -(2 * g) if p.upside_down else (2 * g) > p.y_vel
    if p.upside_down:
        return p.y_vel > g + g
    return (g + g) > p.y_vel


def flip_gravity(p: Player, upside_down: bool) -> None:
    """PlayerObject::flipGravity (0x14039a1d0): only acts when gravity actually changes (§10.2)."""
    if p.upside_down == upside_down:
        return
    p.upside_down = upside_down
    p.last_flip_time = p.time
    p.on_ground = False
    p.y_vel *= 0.5


def update_jump(p: Player, dt: float) -> None:
    """PlayerObject::updateJump (0x14038b900), classic (non-platformer) play (§5)."""
    jump_accel, g, _ = p.constants()
    if p.mode in FIXED_GRAVITY_MODES:
        g = VEHICLE_GRAVITY
    g0 = g
    if p.gravity_mod != 1.0:
        g *= p.gravity_mod
    sz = 0.8 if p.mini else 1.0
    s = p.s

    if p.mode in FLY_MODES:
        if p.mini:
            sz = 0.85
        hi, lo = 8.0 / sz, -6.4 / sz
        v = p.y_vel
        if (s > 0 and ((0 <= v < hi) or (lo < v <= 0))) or (s < 0 and ((-hi < v <= 0) or (0 <= v < -lo))):
            p.outside_clamp = False
        c = 0.8
        if p.mode == "ship":
            k = 0.8
            if p.held and not p.outside_clamp:
                k = -1.0
            elif p.outside_clamp and (p.y_vel < 0 if s > 0 else p.y_vel > 0):
                k = -1.0
            if not p.held and not is_falling(p):
                k = 1.2
            gg = g if k >= 0 else g0
            m = 0.5 if (p.held and is_falling(p)) else 0.4
            p.y_vel = set_y_vel(p.y_vel - s * dt * gg * k * m / sz)
        elif p.mode == "ufo":
            if p.buffer and p.held:
                p.buffer = False
                t = (8.0 if p.mini else 7.0) * sz
                if (p.y_vel < t) if s > 0 else (p.y_vel > -t):
                    p.y_vel = set_y_vel(s * t)
            k = 0.8 if is_falling(p) else 1.2
            p.y_vel = set_y_vel(p.y_vel - s * dt * g * k * 0.5 / sz)
        elif p.mode == "swing":
            if p.buffer and p.held:
                p.buffer = False
                v0 = p.y_vel
                flip_gravity(p, not p.upside_down)
                p.y_vel = set_y_vel(v0 * 0.8)
            sz, c = 1.0, 1.0
            p.y_vel = set_y_vel(p.y_vel - p.s * dt * g * (0.6 if p.mini else 0.4))
        else:  # wave: yVel is cosmetic, motion comes from dx (§3)
            p.y_vel = s * p.speed * p.constants()[2] * (1 if p.held else -1)
        if p.held and p.mode != "wave":
            p.grounded = False
        if not p.outside_clamp and p.mode != "wave":
            if p.s > 0:
                p.y_vel = set_y_vel(min(max(p.y_vel, -8.0 * c / sz), 8.0 / sz))
            else:
                p.y_vel = set_y_vel(min(max(p.y_vel, -8.0 / sz), 8.0 * c / sz))
        if is_falling(p):
            p.is_rising = False
        return

    gm = GROUND_GRAVITY_MULT[p.mode]
    press = (p.buffer and p.held) if p.mode == "robot" else p.held
    if press and p.on_ground and not (p.is_dashing and p.mode != "spider"):
        if p.mode == "spider":
            if not p.is_dashing:
                p.spider_pending = True  # spiderTestJump runs in move()'s hook, before integration (spec §16.1)
            return
        p.is_rising = True
        p.grounded = p.on_ground = p.buffer = p.jump_released = False
        p.robot_boost = 0.0
        v = jump_accel * (0.5 if p.mode == "robot" else 1.0)
        p.y_vel = set_y_vel(s * v * sz)
        if (p.was_on_slope or p.on_slope) and s * p.slope_y_vel > 0:         # jump from a slope (spec §21.6)
            cap = _f32(p.y_vel * 1.4)
            t = p.time - p.slope_enter_time
            f = max(0.4, 10 * t) if t < 0.1 else 1.0
            # spec §21.6 reads "robot ? slopeYVel : slopeYVel*f"; the BRY trace (ball, 0.02 s on the slope) needs the full
            # bonus, so the ramp-in applies to the robot only
            b = p.slope_y_vel * f if p.mode == "robot" else p.slope_y_vel
            p.y_vel = set_y_vel(p.y_vel + 0.25 * b)
            p.y_vel = set_y_vel(min(p.y_vel, cap) if s > 0 else max(p.y_vel, cap))
        if p.mode == "ball":
            flip_gravity(p, not p.upside_down)
            p.y_vel *= 0.6
            p.held = False
        return
    if p.is_rising:
        if p.mode == "robot" and p.held and not p.jump_released and p.robot_boost < 1.5:
            p.robot_boost += dt * 0.1
            p.y_vel = set_y_vel(p.y_vel + s * dt * g * gm)
        p.y_vel = set_y_vel(p.y_vel - s * dt * g * gm)
        if is_falling(p):
            p.is_rising = False
        return
    if is_falling(p):
        p.on_ground = False
    p.y_vel = set_y_vel(p.y_vel - s * dt * g * gm)
    p.y_vel = set_y_vel(max(p.y_vel, -15.0) if s > 0 else min(p.y_vel, 15.0))


def move(p: Player, spider_hook=None) -> None:
    """PlayerObject::update (0x140388d80), minus the bookkeeping handled by the caller (§3, §18.3)."""
    p.y_vel = max(-1000.0, min(1000.0, p.y_vel))
    p.prev_x, p.prev_y = p.x, p.y
    update_jump(p, DTJ)
    if p.spider_pending and spider_hook is not None:
        p.spider_pending = False
        spider_hook(p)
    if p.is_dashing:
        p.y_vel = 0.0
    dx = p.speed * p.constants()[2] * DT60
    if p.is_dashing:
        dy = p.dash_slope * dx
    elif p.mode == "wave":
        dy = abs(dx) * p.s * (1 if p.held else -1)
        if p.mini:
            dy *= 2
    else:
        dy = DTJ * p.y_vel
    p.x += dx
    p.y += dy
    p.ring_held_snap, p.ring_buf_snap = p.held, p.buffer
    p.used_normal_ring = False
