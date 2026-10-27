"""One physics step with collisions, following re/physics_spec.md (§1 step order, §6-§12 collisions).

Simulator.step(state, hold) -> new state: input, movement, level floor / corridor clamp, portals / pads / orbs,
solids, hazards, squish. States are immutable from the caller's point of view (each step copies).

Not simulated yet (recorded in Simulator.unsupported with the first x where they were touched): dual mode
(spec §20). Teleports, dash orbs, the spider teleport and slopes follow spec §16-§18 and §21.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .level import Level, PhysObject, f32
from .physics import (BOUNDED_MODES, CORRIDOR_HEIGHT, FLOOR_Y, FLY_MODES, STEP_SECONDS, Player, flip_gravity,
                      move, set_y_vel, update_jump)
from .triggers import x_step

MODE_PORTALS = {"CubePortal": "cube", "ShipPortal": "ship", "BallPortal": "ball", "UfoPortal": "ufo",
                "WavePortal": "wave", "RobotPortal": "robot", "SpiderPortal": "spider", "SwingPortal": "swing"}
GRAVITY_PORTALS = {"InverseGravityPortal", "NormalGravityPortal", "GravityTogglePortal"}
PADS = {"YellowJumpPad", "PinkJumpPad", "RedJumpPad"}
RINGS = {"YellowJumpRing", "PinkJumpRing", "RedJumpRing", "GravityRing", "GreenRing", "DropRing", "CustomRing",
         "SpiderOrb", "DashRing", "GravityDashRing", "TeleportOrb"}
UNSUPPORTED_TYPES = {"DualPortal": "dual"}
FLY4 = frozenset({"ship", "ufo", "wave", "swing"})
FLYISH5 = FLY4 | {"ball"}
KEEP_BUFFER_RINGS = {"CustomRing", "SpiderOrb", "DashRing", "GravityDashRing"}   # spec §12.1
BROAD_PHASE_MARGIN = 45.0


def pad_mod(kind: str, mode: str, mini: bool) -> float:
    """GJBaseGameLayer::bumpPlayer mod table (§11)."""
    if kind == "YellowJumpPad":
        return 1.0
    if kind == "PinkJumpPad":
        return {"ship": 0.35, "ufo": 0.40, "ball": 0.70, "spider": 0.70}.get(mode, 0.65)
    if mode == "ship":
        return 0.95 if mini else 0.63
    if mode == "ufo":
        return 0.98 if mini else 0.60
    return 1.25


def ring_k(kind: str, mode: str, mini: bool) -> float:
    """PlayerObject::ringJump velocity factors (§12.2)."""
    if kind == "YellowJumpRing":
        return 0.9 if mode == "robot" else 1.0
    if kind == "PinkJumpRing":
        return {"ship": 0.37, "ufo": 0.42, "ball": 0.77}.get(mode, 0.72)
    if kind == "RedJumpRing":
        if mode == "ship":
            return 1.4 if mini else 1.0
        if mode == "ufo":
            return 1.36 if mini else 1.02
        return {"ball": 1.34, "robot": 1.28, "spider": 1.34}.get(mode, 1.38)
    if kind == "GravityRing":
        return 0.8
    if kind == "GreenRing":
        return 0.7 if mode == "ship" else 1.0
    return 1.0


@dataclass(slots=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float


def player_rect(p: Player, inner: bool = False) -> Rect:
    """Outer rect (§2.1) or the 0.3-scale inner rect used for solid deaths, which ignores vehicleSize (§2.2)."""
    w = p.width / p.vehicle_size * 0.3 if inner else p.width
    return Rect(p.x - w / 2, p.y - w / 2, p.x + w / 2, p.y + w / 2)


def overlaps(r: Rect, o: PhysObject) -> bool:
    return r.x0 <= o.x1 and o.x0 <= r.x1 and r.y0 <= o.y1 and o.y0 <= r.y1


def _sat_overlap(r: Rect, corners: tuple) -> bool:
    """Separating-axis test between an axis-aligned rect and a convex quad."""
    rect_pts = ((r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1))
    axes = [(1.0, 0.0), (0.0, 1.0)]
    for i in range(4):
        (ax, ay), (bx, by) = corners[i], corners[(i + 1) % 4]
        axes.append((-(by - ay), bx - ax))
    for nx, ny in axes:
        a = [px * nx + py * ny for px, py in rect_pts]
        b = [px * nx + py * ny for px, py in corners]
        if max(a) < min(b) or max(b) < min(a):
            return False
    return True


def object_hit(r: Rect, o: PhysObject, fix_radius: bool = False) -> bool:
    if o.radius:
        cx, cy = o.x, o.y
        if fix_radius:
            # playerCircleCollision with fixRadiusCollision (§9): player centre within half width + radius.
            return math.hypot((r.x0 + r.x1) / 2 - cx, (r.y0 + r.y1) / 2 - cy) <= (r.x1 - r.x0) / 2 + o.radius
        # Legacy playerCircleCollision (§9): centre inside the rect, or a rect corner within the radius.
        if r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1:
            return True
        return any(math.hypot(px - cx, py - cy) < o.radius
                   for px, py in ((r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)))
    if not overlaps(r, o):
        return False
    return _sat_overlap(r, o.obb) if o.obb else True


def is_flipped(o: PhysObject) -> bool:
    """GameObject::isFlipped (0x1401a1910): ~180 degree rotation XOR flipY (spec §16.1)."""
    r = o.rotation % 360.0
    upside = (abs(r - 180.0) < 1e-6) if r % 90.0 == 0.0 else (91.0 <= r <= 269.0)
    return upside != o.flip_y


def wrap180(a: float) -> float:
    a = math.fmod(a + 180.0, 360.0)
    if a <= 0.0:
        a += 360.0
    return a - 180.0


def hit_ground(p: Player) -> None:
    """PlayerObject::hitGround (0x14039bf30), §8.1."""
    moving_away = p.s * p.y_vel > 5
    p.y_vel = 0.0
    p.grounded = True
    if not moving_away:
        p.on_ground = True


@dataclass
class StepContacts:
    ceil: float | None = None
    floor: float | None = None
    top_uid: int = -1
    bottom_uid: int = -1
    head_uids: set = field(default_factory=set)
    feet_uids: set = field(default_factory=set)

    def head(self, y: float, uid: int) -> None:
        self.ceil = y if self.ceil is None else min(self.ceil, y)
        self.top_uid = uid
        self.head_uids.add(uid)

    def feet(self, y: float, uid: int) -> None:
        self.floor = y if self.floor is None else max(self.floor, y)
        self.bottom_uid = uid
        self.feet_uids.add(uid)


@dataclass
class Simulator:
    level: Level
    unsupported: dict[str, float] = field(default_factory=dict)

    # ----------------------------------------------------------------------------------------------------- setup
    def initial_state(self) -> Player:
        lvl = self.level
        p = Player(mode=lvl.start_mode, mini=lvl.start_mini, speed=lvl.start_speed, upside_down=lvl.start_upside_down)
        # Verified against in-game traces: the player always spawns at y = 105 (any mode) and not on the
        # ground, and the "previous corridor top" is unset on the first step, so that step's clamp sees dMax != 0.
        p.y = p.prev_y = lvl.spawn_y
        p.on_ground = p.grounded = False
        p.fix_gravity_bug = lvl.fix_gravity_bug
        p.speed_log = ((0, p.x, x_step(p.speed)),)
        if p.mode in BOUNDED_MODES:
            p.corridor = self._corridor(p.mode, p.y)
            p.corridor_prev_max = 0.0
        return p

    def near(self, p: Player, x0: float, x1: float) -> list[PhysObject]:
        """Static objects from the index plus trigger-driven ones where the trigger timeline puts them."""
        static = self.level.query(x0, x1)
        engine = self.level.engine
        if engine is None or not engine.sets:
            return static
        return static + engine.timeline(p.speed_log).objects_near(p.tick, x0, x1)

    def _note(self, reason: str, x: float) -> None:
        if reason not in self.unsupported or x < self.unsupported[reason]:
            self.unsupported[reason] = x

    @staticmethod
    def _floor_y(p: Player) -> float:
        return FLOOR_Y + p.bounds_height / 2 * p.vehicle_size

    @staticmethod
    def _corridor(mode: str, portal_y: float) -> tuple[float, float]:
        h = CORRIDOR_HEIGHT[mode]
        bottom = max(FLOOR_Y, math.floor((portal_y - h / 2) / 30.0) * 30.0)
        return (bottom, bottom + h)

    # ------------------------------------------------------------------------------------------------------ step
    def step(self, state: Player, hold: bool) -> Player:
        p = state.copy()
        p.time += STEP_SECONDS
        if p.pending_speed is not None:
            p.speed, p.pending_speed = p.pending_speed, None
            p.speed_log = p.speed_log + ((p.tick, p.x, x_step(p.speed)),)
        press_used: set[int] = set()
        if hold and not p.held:
            self._push_button(p, press_used)
        elif not hold and p.held:
            p.held = p.buffer = False
            p.jump_released = True
            p.is_dashing = False                     # releaseButton ends a dash (spec §18.4)
        move(p, self._spider_teleport)
        p.tick += 1
        self._check_collisions(p, press_used)
        if not p.dead:
            self._post_slope(p)
        if p.activated:
            cutoff = p.x - 120.0
            objs = self.level.objects
            p.activated = frozenset(i for i in p.activated if objs[i].x1 >= cutoff)
        return p

    def _push_button(self, p: Player, press_used: set[int]) -> None:
        """PlayerObject::pushButton (0x140397f40) + ring handling (§12.1)."""
        p.held = p.buffer = True
        p.ring_held_snap = p.ring_buf_snap = True
        p.used_normal_ring = False
        rings = [self.level.objects[i] for i in p.touched_rings if i not in p.activated]
        if rings:
            for ring in sorted(rings, key=lambda o: o.type != "CustomRing"):
                self._ring_jump(p, ring, press_used)
        elif p.mode == "ball" or (p.mode not in FLY_MODES and p.held and p.on_ground):
            update_jump(p, 0.0)

    # ------------------------------------------------------------------------------------------------ collisions
    def _check_collisions(self, p: Player, press_used: set[int]) -> None:
        """GJBaseGameLayer::checkCollisions (0x1402137f0), §6."""
        oob_prev, oob = p.oob_prev, False
        teleported, p.teleported = p.teleported, False
        p.was_on_slope, p.on_slope = p.on_slope, False
        c = StepContacts()
        floor_y = self._floor_y(p)
        bounded = p.mode in BOUNDED_MODES and not p.free_mode
        if p.y >= floor_y or bounded:
            v11 = (1.0 - p.vehicle_size) * p.bounds_height / 2
            if p.y > v11 + self.level.max_gameplay_y:
                oob = True
        else:
            if p.upside_down:
                if p.last_flip_time and p.time - p.last_flip_time < 0.1:
                    p.y = floor_y
                    hit_ground(p)
                    p.grounded = False
                    p.touched_rings = ()
                    self._finish(p, c, oob, oob_prev)
                    return
                oob = True
            elif not p.is_rising:
                p.y = floor_y
                hit_ground(p)
                c.feet(FLOOR_Y, -2)
        if bounded and p.corridor and not teleported:
            self._clamp_corridor(p, c)

        r = player_rect(p)
        solids: list[PhysObject] = []
        hazards: list[PhysObject] = []
        touched: list[int] = []
        for o in self.near(p, r.x0 - BROAD_PHASE_MARGIN, r.x1 + BROAD_PHASE_MARGIN):
            cat = o.category
            if cat == "solid":
                if o.index not in p.activated:
                    solids.append(o)
                continue
            if cat == "hazard":
                hazards.append(o)
                continue
            if o.index in p.activated or not object_hit(r, o):
                continue
            if cat == "slope":
                self._slope(p, o, c)
                if p.dead:
                    return
                r = player_rect(p)
                continue
            if o.type in UNSUPPORTED_TYPES:
                self._note(UNSUPPORTED_TYPES[o.type], o.x)
                continue
            self._activate(p, o, c, touched, press_used)
            r = player_rect(p)                       # teleports and spider jumps move the player
        p.touched_rings = tuple(touched)

        r = player_rect(p)
        for o in sorted(solids, key=lambda o: -o.index):
            if overlaps(r, o):
                self._collide_solid(p, o, c)
                if p.dead:
                    return
                r = player_rect(p)                   # only a collision moves the player
        r = player_rect(p)
        fix_radius = self.level.fix_radius_collision
        for o in hazards:
            if object_hit(r, o, fix_radius):
                p.dead, p.killer = True, o.index
                return
        self._finish(p, c, oob, oob_prev)

    def _finish(self, p: Player, c: StepContacts, oob: bool, oob_prev: bool) -> None:
        """postCollision squish checks (§6.7) and the two-step out-of-bounds rule (§6.3)."""
        if c.ceil is not None and c.floor is not None and abs(c.ceil - c.floor) < 0.7 * p.width:
            p.dead, p.killer = True, -2
        if c.head_uids & c.feet_uids:
            p.dead, p.killer = True, -2
        if oob and oob_prev:
            p.dead, p.killer = True, -3
        p.oob_prev = oob
        p.prev_top_uid, p.prev_bottom_uid = c.top_uid, c.bottom_uid

    def _clamp_corridor(self, p: Player, c: StepContacts) -> None:
        """Bounded-mode floor/ceiling (§6.3)."""
        min_y, max_y = p.corridor
        d_max = max_y - p.corridor_prev_max
        p.corridor_prev_max = max_y
        half = p.bounds_height / 2 * p.vehicle_size
        ceil_lim, floor_lim = max_y - half, min_y + half
        ship_held = p.mode == "ship" and p.held
        old = p.y_vel
        if p.y <= ceil_lim:
            if p.y >= floor_lim:
                return
            restore = (not p.upside_down) and (p.mode not in ("ship", "wave") or not p.buffer) and d_max != 0
            p.was_on_slope = False
            p.y = floor_lim
            if p.mode not in ("ship", "swing") or p.y_vel < 0:
                hit_ground(p)
            c.feet(min_y, -2)
            if (restore and p.y_vel > old) or (p.upside_down and ship_held and d_max < 0):
                p.y_vel = old
        else:
            restore = p.upside_down and (p.mode not in ("ship", "wave") or not p.buffer) and d_max != 0
            p.was_on_slope = False
            p.y = ceil_lim
            if p.y_vel > 0:
                hit_ground(p)
            c.head(max_y, -3)
            if (restore and old > p.y_vel) or (not p.upside_down and ship_held and d_max > 0):
                p.y_vel = set_y_vel(old)

    def _spider_teleport(self, p: Player, c: StepContacts | None = None, dynamic: bool = False) -> None:
        """spiderTestJumpInternal (0x1403943f0), classic forward play (spec §16.2-§16.4)."""
        r = player_rect(p)
        h = r.y1 - r.y0
        px, py, s = p.x, p.y, p.s
        vs = p.vehicle_size
        if p.mode in BOUNDED_MODES and p.corridor and not p.free_mode:
            bottom, top = p.corridor
        else:
            bottom, top = max(FLOOR_Y, py - 3000.0), py + 3000.0
        if s > 0:
            ay = (r.y1 - 2.0, top) if not dynamic else (r.y1 + 2.0, top + 4.0)
        else:
            ay = (bottom, r.y0 + 2.0) if not dynamic else (bottom + 4.0, r.y0 + 6.0)
        area = Rect(px, ay[0], r.x1 + 1.0, ay[1])
        strip = Rect(px - 4.0 * vs, ay[0], px + 4.0 * vs, ay[1])
        nearby = self.near(p, min(area.x0, strip.x0) - 60.0, max(area.x1, strip.x1) + 60.0)
        solids = [o for o in nearby if o.category in ("solid", "slope") and o.index not in p.activated
                  and overlaps(area, o)]
        hazards = [o for o in nearby if o.category == "hazard" and object_hit(strip, o)]
        if not solids and hazards and not dynamic:
            return self._spider_teleport(p, c, dynamic=True)
        if s > 0:
            solids.sort(key=lambda o: int(o.y0 * 10))
            hazards.sort(key=lambda o: int(o.y0 * 10))
        else:
            solids.sort(key=lambda o: -int(o.y1 * 10))
            hazards.sort(key=lambda o: -int(o.y1 * 10))
        target = 0.0
        for i, o in enumerate(solids):
            t = o.y0 - h / 2 if s > 0 else o.y1 + h / 2
            if (s > 0 and t < py - 10) or (s < 0 and t > py + 10):
                continue
            if o.x0 <= px:
                target = t
                break
            moved = Rect(r.x0, t - h / 2, r.x1, t + h / 2)
            if not any(j != i and moved.x0 <= q.x1 and q.x0 <= moved.x1 and moved.y0 <= q.y1 - 2 and q.y0 + 2 <= moved.y1
                       for j, q in enumerate(solids)):
                target = t
                break
        if not solids and hazards:
            target = hazards[0].y
        else:
            if target == 0.0:
                target = top - h / 2 if s > 0 else bottom + h / 2
            if hazards:
                hy = hazards[0].y
                if (s > 0 and hy < target) or (s < 0 and hy > target):
                    if not dynamic:
                        return self._spider_teleport(p, c, dynamic=True)
                    target = hy
        p.y = target
        flip_gravity(p, not p.upside_down)
        p.y_vel = set_y_vel(1.0 if p.upside_down else -1.0)
        p.held = False
        p.spider_tp_time = p.time
        p.prev_top_uid = p.prev_bottom_uid = -1
        if c is not None:
            c.head_uids.clear()
            c.feet_uids.clear()

    def _teleport(self, p: Player, o: PhysObject) -> None:
        """GJBaseGameLayer::teleportPlayer (0x14020fdb0), spec §17.3."""
        params = o.params or {}
        num = lambda k, d=0.0: float(params.get(k, d) or d)  # noqa: E731
        p.teleported = True
        dest = None
        if o.type == "TeleportPortal":
            dest = (p.x, o.y + num("54"))
        elif num("51") > 0:
            group = int(num("51"))
            targets = [q for q in self.level.objects if group in q.groups]
            if targets:
                dest = (targets[0].x, targets[0].y)
                self._note("teleport to group (x jump)", o.x)
        if dest is not None:
            dx, dy = dest
            if params.get("351") == "1":                         # save offset
                dx -= o.x - p.x
                dy -= o.y - p.y
            if params.get("352") == "1":
                dx = p.x
            if params.get("353") == "1":
                dy = p.y
            p.x, p.y = dx, dy
            p.grounded = False
        gravity = int(num("354"))
        if gravity == 1:
            flip_gravity(p, False)
        elif gravity == 2:
            flip_gravity(p, True)
        elif gravity == 3:
            flip_gravity(p, not p.upside_down)
        angle = math.radians(180.0 - o.rotation + (180.0 if o.flip_x else 0.0))
        if params.get("347") == "1":                             # redirect force
            length = abs(p.y_vel) * (num("350", 1.0) or 1.0)
            if num("349") > 0:
                length = min(length, num("349"))
            if num("348") > 0:
                length = max(length, num("348"))
            p.y_vel = length * math.sin(angle)
            p.outside_clamp = True
        elif params.get("345") == "1":                           # static force
            force = num("346")
            additive = params.get("443") == "1"
            if force == 0.0 and not additive:
                p.outside_clamp = False
                p.y_vel = 0.0
            else:
                p.outside_clamp = True
                p.y_vel = force * math.sin(angle) + (p.y_vel if additive else 0.0)

    def _start_dash(self, p: Player, o: PhysObject) -> None:
        """PlayerObject::startDashing (0x140395910), classic forward play (spec §18.2)."""
        if p.is_dashing:
            return
        p.is_dashing = True
        a = o.rotation + (180.0 if o.flip_x else 0.0)
        t = wrap180(math.fmod(-a, 360.0))
        if abs(t) > 90.0:
            t = wrap180((180.0 if t > 0 else -180.0) - t)
        t = max(-70.0, min(70.0, t))
        rad = math.radians(t)
        p.dash_slope = math.sin(rad) / abs(math.cos(rad))

    # ----------------------------------------------------------------------------------------- special objects
    def _consume(self, p: Player, o: PhysObject) -> None:
        if not o.multi_activate:
            p.activated = p.activated | {o.index}

    def _activate(self, p: Player, o: PhysObject, c: StepContacts, touched: list[int], press_used: set[int]) -> None:
        t = o.type
        if t in GRAVITY_PORTALS:
            target = {"InverseGravityPortal": True, "NormalGravityPortal": False}.get(t, not p.upside_down)
            self._consume(p, o)
            if target != p.upside_down:
                flip_gravity(p, target)
                c.ceil = c.floor = None
                c.head_uids.clear()
                c.feet_uids.clear()
        elif t in MODE_PORTALS:
            self._consume(p, o)
            p.free_mode = o.free_mode
            self._switch_mode(p, MODE_PORTALS[t], o.y)
        elif t in PADS:
            self._consume(p, o)
            self._propel(p, pad_mod(t, p.mode, p.mini))
            p.outside_clamp = t == "RedJumpPad"
            p.jump_released = True
        elif t == "GravityPad":
            pad_flipped = is_flipped(o)
            if p.upside_down != (not pad_flipped):
                self._consume(p, o)
                self._propel(p, 0.8)
                flip_gravity(p, not pad_flipped)
                p.jump_released = True
        elif t in RINGS:
            touched.append(o.index)
            if p.mode not in FLY_MODES:
                self._ring_jump(p, o, press_used)
        elif t == "TeleportPortal":
            self._consume(p, o)
            self._teleport(p, o)
        elif t == "SpiderPad":
            self._consume(p, o)
            p.jump_released = True
            if is_flipped(o) != p.upside_down:
                flip_gravity(p, not p.upside_down)
            self._spider_teleport(p, c)
        elif t in ("RegularSizePortal", "MiniSizePortal"):
            self._consume(p, o)
            p.mini = t == "MiniSizePortal"
        elif o.category == "speed":
            self._consume(p, o)
            p.pending_speed = o.speed

    def _switch_mode(self, p: Player, mode: str, portal_y: float) -> None:
        """switchedToMode + toggle*Mode (§10.1) and the corridor from playerWillSwitchMode (§6.3)."""
        if mode != p.mode:
            if p.mode in FLY_MODES:
                p.y_vel *= 0.5
                p.on_ground = p.grounded = False
            p.mode = mode
            if mode in FLY_MODES:
                p.y_vel *= 0.5
                p.on_ground = p.grounded = False
            if mode in ("robot", "spider"):
                p.robot_boost = 1.5
            p.last_mode_change_time = p.time
        p.corridor = self._corridor(mode, portal_y) if mode in BOUNDED_MODES and not p.free_mode else None

    def _propel(self, p: Player, mod: float) -> None:
        """PlayerObject::propellPlayer (§11)."""
        p.is_rising = True
        p.grounded = p.on_ground = False
        p.y_vel = set_y_vel(p.s * mod * 16.0 * (0.8 if p.mini else 1.0))
        if p.mode in ("ball", "spider", "swing"):
            p.y_vel *= 0.6

    def _ring_jump(self, p: Player, o: PhysObject, press_used: set[int]) -> None:
        """PlayerObject::ringJump (0x140398c00), §12."""
        t = o.type
        if p.dead or p.is_dashing or o.index in press_used or not p.ring_buf_snap or not p.ring_held_snap:
            return
        if t == "CustomRing" or p.used_normal_ring:
            return
        if t not in KEEP_BUFFER_RINGS:
            p.buffer = False
        if t == "SpiderOrb":
            if is_flipped(o) != p.upside_down:
                flip_gravity(p, not p.upside_down)
            self._spider_teleport(p)
        elif t in ("DashRing", "GravityDashRing"):
            if t == "GravityDashRing":
                flip_gravity(p, not p.upside_down)
            self._start_dash(p, o)
        elif t == "TeleportOrb":
            self._teleport(p, o)
        elif t == "DropRing":
            v = -14.0 if p.mode in FLY_MODES else -15.0
            if p.mode == "ufo":
                v *= 0.8
            elif p.mode == "spider":
                v *= 1.1
            p.y_vel = set_y_vel(v * p.s)
            p.outside_clamp = True
            if p.mode in ("ball", "swing"):
                p.held = False
        else:
            if t == "GreenRing":
                flip_gravity(p, not p.upside_down)
            jump_accel = p.constants()[0]
            p.is_rising = True
            p.grounded = p.on_ground = False
            p.y_vel = set_y_vel(p.s * jump_accel * ring_k(t, p.mode, p.mini) * (0.8 if p.mini else 1.0))
            if t == "RedJumpRing":
                p.outside_clamp = True
            if p.mode in ("ball", "spider"):
                p.y_vel *= 0.7
                p.held = False
            elif p.mode == "swing":
                p.y_vel *= 0.6
                p.held = False
            if t == "GravityRing":
                flip_gravity(p, not p.upside_down)
        p.used_normal_ring = True
        press_used.add(o.index)
        self._consume(p, o)

    # ---------------------------------------------------------------------------------------------------- slopes
    @staticmethod
    def slope_y(o: PhysObject, x: float) -> float:
        """slopeYPos (spec §21.2): the surface line at x, extrapolated beyond the rect."""
        k = (o.y1 - o.y0) / (o.x1 - o.x0)
        uphill = o.slope_dir in (0, 3, 6, 7)
        if x > o.x0:
            d = (x - o.x1) * k
            return o.y1 + d if uphill else o.y0 - d
        d = (o.x0 - x) * k
        return o.y0 - d if uphill else o.y1 + d

    def _pre_slope(self, p: Player, o: PhysObject, c: StepContacts) -> bool:
        """preSlopeCollision (spec §21.3): True when a flat side acted like a block and the surface is skipped."""
        if o.index == p.slope_uid:
            return False
        ceiling = o.slope_dir in (1, 3, 5, 6)
        r = player_rect(p)
        if o.slope_dir in (2, 3, 4, 6) and p.x < o.x0:                       # the vertical wall on the left
            wall = replace(o, x0=o.x0, x1=o.x0 + 1.0, category="solid", type="Solid", passable=False)
            if overlaps(r, wall):
                self._collide_solid(p, wall, c)
                return True
        if (p.y >= o.y1) if ceiling else (p.y <= o.y0 + 1):                 # the flat edge
            y0 = o.y1 - 1.0 if ceiling else o.y0
            edge = replace(o, y0=y0, y1=y0 + 1.0, category="solid", type="Solid", passable=False)
            if overlaps(r, edge):
                self._collide_solid(p, edge, c)
                return True
        return False

    def _slope(self, p: Player, o: PhysObject, c: StepContacts) -> None:
        """collidedWithSlopeInternal (spec §21.4), classic forward play."""
        if self._pre_slope(p, o, c) or p.dead:
            return
        uphill, ceil = o.slope_dir in (0, 3, 6, 7), o.slope_dir in (1, 3, 5, 6)
        descending = not uphill
        if p.upside_down:
            descending = not descending
        s, mode = p.s, p.mode
        theta = o.slope_theta
        ow, oh = o.x1 - o.x0, o.y1 - o.y0
        spd = p.speed * p.constants()[2]
        vs = f32(oh / (ow / spd))
        if p.was_on_slope and s * p.slope_y_vel > 0:
            if descending and p.slope_ceil == ceil == p.upside_down:
                return
            if p.slope_vs > vs:
                return
            if p.slope_clamped and o.index == p.slope_uid and not descending:
                return                      # crested the end of a rising slope last step: leave it (launch)
        r = player_rect(p)
        h = r.y1 - r.y0
        px, py = p.x, p.y
        tol = (4.0 if p.was_on_slope else 1.0) if descending else 0.0
        if p.was_on_slope:
            cos_a = math.cos(p.slope_angle)
            probe = py + (h / 2) / cos_a + tol if p.upside_down else py - (h / 2) / cos_a - tol
            if (s > 0 and probe > o.y1) or (s < 0 and probe < o.y0):
                return
        elif not (r.x0 <= o.x1 and o.x0 <= r.x1 and r.y0 <= o.y1 - 1 and o.y0 + 1 <= r.y1):
            return
        hc = (h / math.cos(theta)) / 2
        trans = p.was_on_slope and p.slope_uid != o.index and p.slope_ceil != ceil
        lift = 20 * p.vehicle_size if trans else 0.0
        ys = self.slope_y(o, px)
        if not ceil:
            raw = ys + hc - lift
            target = min(max(raw, o.y0), o.y1 + h / 2)
            clamped = raw > o.y1 + h / 2
        else:
            raw = ys - hc + lift
            target = min(max(raw, o.y0 - h / 2), o.y1)
            clamped = raw < o.y0 - h / 2
        feet_side = ceil == p.upside_down
        fly_held = False
        if feet_side:
            snap_tol = (descending and not trans and not p.on_slope
                        and ((not p.held) if mode == "ship" else (not p.is_rising or mode in ("ufo", "wave", "swing"))))
            if s > 0:
                hit = py < target or (snap_tol and py < target + tol and not (mode == "ufo" and p.y_vel > 0))
            else:
                hit = py > target or (snap_tol and py > target - tol and not (mode == "ufo" and p.y_vel < 0))
            fly_held = p.held and mode in FLY4 and not descending
        else:
            tol2 = 0.0 if descending else ((1.0 + p.was_on_slope) if mode in FLY4 and p.held else 0.0)
            ok = not trans and (mode not in FLY4 or not descending) and not p.on_slope
            hit = (py > target or (ok and py > target - tol2)) if s > 0 else (py < target or (ok and py < target + tol2))
            if hit and mode in ("cube", "robot", "spider"):
                if p.time - p.last_mode_change_time < 0.1 or p.time - p.last_flip_time < 0.1:
                    p.y = target
                    p.y_vel = min(p.y_vel, -2.0) if s > 0 else max(p.y_vel, 2.0)
                    p.on_ground = p.grounded = False
                    return
                if (py - 2 <= target) if s > 0 else (py + 2 >= target):
                    p.on_ground = p.grounded = False
                    return
                p.dead, p.killer = True, o.index
                return
        if not hit:
            return
        if trans or mode == "wave":                                         # spec §21.4 death on contact
            p.dead, p.killer = True, o.index
            return
        if descending and mode == "wave" and p.held and ceil == p.upside_down:
            return
        p.slope_uid, p.slope_ceil, p.slope_desc, p.slope_clamped = o.index, ceil, descending, clamped
        p.slope_angle = (s * theta) * (1.0 if descending else -1.0)
        p.y = target
        if not p.was_on_slope:
            p.slope_enter_time = p.time
        p.on_slope = True
        if feet_side and not fly_held:
            v = p.y_vel
            hit_ground(p)
            if s * v > 5:
                p.y_vel = set_y_vel(v)
            c.feet(target - s * h / 2, o.index) if s > 0 else c.head(target - s * h / 2, o.index)
        if not feet_side and not (not p.held and mode in FLYISH5 and descending and p.was_on_slope):
            p.y_vel = min(p.y_vel, 0.0) if s > 0 else max(p.y_vel, 0.0)
            p.on_ground = p.grounded = False
            c.head(target + s * h / 2, o.index) if s > 0 else c.feet(target + s * h / 2, o.index)
        if not feet_side and mode in FLY4 and descending and not p.held:
            p.y_vel = min(p.y_vel, -2.0) if s > 0 else max(p.y_vel, 2.0)
        if fly_held:
            p.y_vel = max(p.y_vel, 2.0) if s > 0 else min(p.y_vel, -2.0)
        p.slope_vs = vs
        p.slope_y_vel = f32(s * (-1.0 if descending else 1.0) * min(1.1, 0.8 / theta) * vs * 1.4)
        if mode in FLYISH5:
            p.slope_y_vel = f32(p.slope_y_vel * 0.75)

    def _post_slope(self, p: Player) -> None:
        """postCollision: leaving a slope launches the player or keeps it falling at the slope's rate (§21.6)."""
        if p.on_slope or not p.was_on_slope:
            return
        s, c, v_slope = p.s, p.slope_ceil, p.slope_y_vel
        launch = (v_slope > 0 and v_slope > p.y_vel and not c) or (v_slope < 0 and p.y_vel > v_slope and c)
        if launch and (not p.is_rising or p.mode in FLYISH5):
            t = p.time - p.slope_enter_time
            v = v_slope * (max(0.4, 10 * t) if t < 0.1 else 1.0)
            if not p.upside_down:
                apply = v < p.y_vel if (c and p.slope_desc) else v > p.y_vel
            else:
                apply = v > p.y_vel if (not c and p.slope_desc) else v < p.y_vel
            if apply:
                p.is_rising = True
                p.grounded = p.on_ground = False
                p.outside_clamp = True
                p.y_vel = set_y_vel(v)
        elif (s > 0 and v_slope < 0) or (s < 0 and v_slope > 0):
            if not p.is_rising and not p.held and not p.buffer and c == p.upside_down:
                p.grounded = p.on_ground = False
                p.y_vel = set_y_vel(-s * p.slope_vs)
        p.slope_vs = p.slope_y_vel = p.slope_angle = 0.0
        p.slope_uid, p.slope_clamped = -1, False

    # ---------------------------------------------------------------------------------------------------- solids
    def _collide_solid(self, p: Player, o: PhysObject, c: StepContacts) -> None:
        """PlayerObject::collidedWithObjectInternal, classic mode (§7)."""
        s = p.s
        h = p.width
        dy = p.y - p.prev_y
        fly = p.mode in FLY_MODES
        tol = 6.0 if fly else 10.0
        feet = p.y - s * h / 2 + s * tol
        feet_prev = feet - dy
        head = p.y + s * h / 2 - s * tol
        head_prev = head - dy
        flyish = fly or p.mode == "ball"
        passable = o.passable
        if s > 0:
            flag71 = o.y0 >= feet or o.y0 >= feet_prev
        else:
            flag71 = feet >= o.y1 or feet_prev >= o.y1

        if p.mode != "wave":
            # A: the object is under the player's world-bottom (landing in normal gravity).
            below = (feet >= o.y1 or feet_prev >= o.y1) if s > 0 else (head >= o.y1 or head_prev >= o.y1)
            if ((not p.upside_down) or (flyish and not passable)) and below:
                if p.y_vel <= 0:
                    p.y = o.y1 + h / 2
                    if flag71:
                        p.y_vel = 0.0
                    else:
                        hit_ground(p)
                    (c.feet if s > 0 else c.head)(o.y1, o.index)
                return
            # B: the object is above the player's world-top (ceiling; landing when upside-down).
            if p.upside_down or (flyish and not passable):
                cmp, cmp_prev = (head, head_prev) if s > 0 else (feet, feet_prev)
                deep = o.y0 < cmp and o.y0 < cmp_prev
                prev_uid = p.prev_top_uid if s > 0 else p.prev_bottom_uid
                if not deep or o.index == prev_uid:
                    if p.y_vel >= 0:
                        p.y = o.y0 - h / 2
                        if flag71:
                            p.y_vel = 0.0
                        else:
                            hit_ground(p)
                        if not ((passable or o.type == "Breakable") and flag71 and s > 0):
                            (c.head if s > 0 else c.feet)(o.y0, o.index)
                    return

        # C: death check with the 0.3-scale inner box (§7C); passable blocks never kill (§15.3).
        if passable:
            return
        inner = player_rect(p, inner=True)
        if not overlaps(inner, o):
            return
        if p.mode == "spider" and inner.x1 > o.x1 and p.time - p.spider_tp_time < 0.04:
            return
        if flag71 and p.last_flip_time and p.time - p.last_flip_time < 0.1:
            p.y = o.y1 + h / 2 if p.upside_down else o.y0 - h / 2
            hit_ground(p)
            p.grounded = False
            return
        if o.type == "Breakable":
            p.activated = p.activated | {o.index}
            return
        p.dead, p.killer = True, o.index

    # ---------------------------------------------------------------------------------------------- search hook
    @staticmethod
    def key(p: Player) -> tuple:
        # Coarse on purpose: merged states keep one exact representative, so any path found is still exact.
        return (round(p.y * 2.0), round(p.y_vel * 4.0), p.mode, p.mini, p.upside_down, p.speed, p.held, p.buffer,
                p.on_ground, p.is_rising, p.outside_clamp, round(p.robot_boost, 2), p.pending_speed, p.corridor,
                p.touched_rings, p.activated, p.oob_prev, p.time - p.last_flip_time < 0.1, p.speed_log,
                p.is_dashing, round(p.dash_slope, 3), p.teleported, p.on_slope, p.slope_uid, p.slope_ceil, p.slope_clamped,
                round(p.slope_y_vel * 100))

    @staticmethod
    def bucket(p: Player) -> tuple:
        """Coarse group for frontier pruning: states of one group behave alike for the next stretch."""
        return p.mode, p.upside_down, p.mini, p.speed, int(p.y // 10)
