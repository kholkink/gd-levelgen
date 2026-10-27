"""Turns a parsed level (header + object dicts) into physics objects with world-space hitboxes.

Coordinates are in-game units: one grid cell is 30 and the floor's top surface is y = 90. Level strings store y
relative to the floor (ground-row objects sit at y = 15), so LEVEL_Y_OFFSET is added on load.
Rotation (key 6) is in degrees, clockwise.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

from ..objects import GAMEMODE_BY_TYPE, SPEED_PORTAL_IDS, ObjectInfo, load_objects

KEY_ID, KEY_X, KEY_Y, KEY_FLIP_X, KEY_FLIP_Y, KEY_ROT = "1", "2", "3", "4", "5", "6"
KEY_SCALE, KEY_SCALE_X, KEY_SCALE_Y = "32", "128", "129"
KEY_GROUPS = "57"
KEY_NO_TOUCH = "121"
KEY_MULTI_ACTIVATE = "99"
KEY_TARGET_GROUP = "51"
KEY_PASSABLE = "134"
TELEPORT_KEYS = ("51", "54", "345", "346", "347", "348", "349", "350", "351", "352", "353", "354", "443")

# Triggers that move, rotate, scale, toggle or animate their target group: objects in those groups are not
# where the static layout puts them (move, toggle, rotate, follow, follow-player-y, scale, area move/rotate/scale,
# advanced follow, keyframe animation). The trigger engine simulates move and toggle (ENGINE_TRIGGER_IDS).
DYNAMIC_TRIGGER_IDS = frozenset({901, 1049, 1346, 1347, 1814, 2067, 3006, 3007, 3008, 3016, 3033})
ENGINE_TRIGGER_IDS = frozenset({901, 1049})
APPROX_TRIGGER_IDS = DYNAMIC_TRIGGER_IDS - ENGINE_TRIGGER_IDS     # rotate/follow/scale/area/keyframe: not simulated

# Level header (kA*) values.
HEADER_GAMEMODES = ["cube", "ship", "ball", "ufo", "wave", "robot", "spider", "swing"]
HEADER_SPEEDS = [0.9, 0.7, 1.1, 1.3, 1.6]       # kA4: normal, slow, fast, faster, fastest -> m_playerSpeed

LEVEL_Y_OFFSET = 90.0
COLLIDABLE = frozenset({"solid", "slope", "hazard", "orb", "pad", "portal_mode", "portal_gravity",
                        "portal_mirror", "portal_size", "portal_dual", "teleport", "speed"})
BUCKET = 60.0
# spec §21.1: (flipX, flipY) -> slope direction for rotation 0, 180, 90 and 270 degrees
SLOPE_DIRS = {(False, False): (0, 3, 4, 5), (False, True): (1, 2, 7, 6), (True, False): (2, 1, 6, 7),
              (True, True): (3, 0, 5, 4)}


def f32(v: float) -> float:
    """Round to a C float (the game keeps slope geometry in floats)."""
    return struct.unpack("f", struct.pack("f", v))[0]


@dataclass(slots=True)
class PhysObject:
    index: int
    id: int
    category: str
    type: str
    x: float                 # object position (hitbox center)
    y: float
    x0: float                # axis-aligned bounds of the hitbox (of the OBB if rotated)
    y0: float
    x1: float
    y1: float
    radius: float | None = None
    obb: tuple | None = None          # 4 corners when rotated by a non-multiple of 90 degrees
    rotation: float = 0.0
    flip_x: bool = False
    flip_y: bool = False
    multi_activate: bool = False
    groups: tuple[int, ...] = ()
    gamemode: str | None = None       # for portal_mode
    speed: float | None = None        # m_playerSpeed for speed portals
    source: int = -1                  # index of the object in the parsed level
    passable: bool = False            # key 134: one-way platform (spec §15.3)
    free_mode: bool = False           # key 111 on mode portals: no corridor (spec §6.3 freeMode)
    params: dict | None = None        # raw keys some objects need at runtime (teleport options, ...)
    dynamic: bool = False             # in a group targeted by a moving/toggling trigger
    slope_dir: int = 0                # spec §21.1 orientation 0..7 (uphill: 0,3,6,7; ceiling slope: 1,3,5,6)
    slope_theta: float = 0.0          # atanf(h / w) of the slope's rect


@dataclass
class Level:
    objects: list[PhysObject]
    start_mode: str = "cube"
    start_mini: bool = False
    start_speed: float = 0.9
    start_upside_down: bool = False
    start_dual: bool = False
    length: float = 0.0
    dynamic_skipped: int = 0
    fix_gravity_bug: bool = False     # kA32 (spec §23.3)
    spawn_y: float = 105.0            # player spawn height (kA36 spawn group, see build_level)
    fix_radius_collision: bool = False  # circle hazards: centre-distance test (spec §9); header kA39 (LevelSettings+463, verified)
    max_gameplay_y: float = 2800.0    # out-of-bounds height (spec §22)
    dynamic_objects: list[PhysObject] = field(default_factory=list)
    engine: object | None = None      # triggers.TriggerEngine when trigger-driven objects are simulated
    _bucket_index: dict[int, list[int]] = field(default_factory=dict, repr=False)
    _query_cache: dict[tuple[int, int], list[PhysObject]] = field(default_factory=dict, repr=False)

    def build_index(self) -> None:
        self._bucket_index.clear()
        self._query_cache.clear()
        for i, o in enumerate(self.objects):
            if o.dynamic and self.engine is not None:
                continue
            for b in range(math.floor(o.x0 / BUCKET), math.floor(o.x1 / BUCKET) + 1):
                self._bucket_index.setdefault(b, []).append(i)
        self.length = max((o.x1 for o in self.objects if o.category != "speed"), default=0.0)

    def query(self, x0: float, x1: float) -> list[PhysObject]:
        """Objects whose bounds may overlap [x0, x1] horizontally (bucket-granular; result is shared, don't mutate)."""
        b0, b1 = math.floor(x0 / BUCKET), math.floor(x1 / BUCKET)
        cached = self._query_cache.get((b0, b1))
        if cached is not None:
            return cached
        seen: set[int] = set()
        for b in range(b0, b1 + 1):
            seen.update(self._bucket_index.get(b, ()))
        out = [self.objects[i] for i in sorted(seen)]
        if len(self._query_cache) > 4096:
            self._query_cache.clear()
        self._query_cache[(b0, b1)] = out
        return out


def _float(obj: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(obj.get(key, default))
    except ValueError:
        return default


def _rotated_corners(cx: float, cy: float, w: float, h: float, degrees: float) -> tuple:
    a = math.radians(-degrees)                     # GD rotates clockwise
    c, s = math.cos(a), math.sin(a)
    pts = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
    return tuple(pts)


def make_phys_object(index: int, obj: dict, info: ObjectInfo) -> PhysObject:
    x, y = _float(obj, KEY_X), _float(obj, KEY_Y) + LEVEL_Y_OFFSET
    uniform = _float(obj, KEY_SCALE, 1.0) or 1.0
    sx = _float(obj, KEY_SCALE_X, uniform) or uniform
    sy = _float(obj, KEY_SCALE_Y, uniform) or uniform
    raw_rot = _float(obj, KEY_ROT)
    if info.category in ("solid", "slope") and raw_rot % 90.0 != 0.0:
        raw_rot = 0.0                 # objectFromVector drops non-90 rotations of solids/slopes (spec §15.2)
    rot = raw_rot % 360.0
    # GameObject::setRotation (0x140197e00) sets the width/height swap flag (+912) only for a raw rotation of exactly
    # +-90 or +-270, so e.g. 450 keeps the unrotated box although it looks like 90 [V]
    swap = raw_rot in (90.0, -90.0, 270.0, -270.0)
    w, h = info.hitbox
    w, h = abs(w * sx), abs(h * sy)
    po = PhysObject(index=index, id=info.id, category=info.category, type=info.type, x=x, y=y,
                    x0=x, y0=y, x1=x, y1=y, rotation=rot,
                    flip_x=obj.get(KEY_FLIP_X) == "1", flip_y=obj.get(KEY_FLIP_Y) == "1",
                    multi_activate=obj.get(KEY_MULTI_ACTIVATE) == "1", passable=obj.get(KEY_PASSABLE) == "1",
                    groups=tuple(int(g) for g in obj.get(KEY_GROUPS, "").split(".") if g.isdigit()))
    if info.radius:
        r = info.radius * max(abs(sx), abs(sy))
        po.radius = r
        po.x0, po.y0, po.x1, po.y1 = x - r, y - r, x + r, y + r
    elif rot % 90.0 == 0.0:
        if swap:
            w, h = h, w
        po.x0, po.y0, po.x1, po.y1 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    else:
        corners = _rotated_corners(x, y, w, h, rot)
        po.obb = corners
        xs, ys = [p[0] for p in corners], [p[1] for p in corners]
        po.x0, po.y0, po.x1, po.y1 = min(xs), min(ys), max(xs), max(ys)
    if info.category == "slope":
        fx, fy = po.flip_x, po.flip_y
        column = {0.0: 0, 180.0: 1, 90.0: 2, 270.0: 3}.get(rot, 0)
        po.slope_dir = SLOPE_DIRS[(fx, fy)][column]
        po.slope_theta = f32(math.atan(f32((po.y1 - po.y0) / (po.x1 - po.x0)))) if po.x1 > po.x0 else 0.0
    if info.type in ("TeleportPortal", "TeleportOrb"):
        po.params = {k: obj[k] for k in TELEPORT_KEYS if k in obj}
    if info.category == "portal_mode":
        po.gamemode = GAMEMODE_BY_TYPE[info.type]
        po.free_mode = obj.get("111") == "1"
    elif info.category == "speed":
        po.speed = {0.5: 0.7, 1.0: 0.9, 2.0: 1.1, 3.0: 1.3, 4.0: 1.6}[SPEED_PORTAL_IDS[info.id]]
    return po


def dynamic_groups(objects: list[dict[str, str]], trigger_ids: frozenset = DYNAMIC_TRIGGER_IDS) -> set[int]:
    groups = set()
    for obj in objects:
        try:
            if int(obj.get(KEY_ID, "0")) in trigger_ids:
                groups.add(int(obj.get(KEY_TARGET_GROUP, "0")))
        except ValueError:
            continue
    groups.discard(0)
    return groups


def build_level(header: dict[str, str], objects: list[dict[str, str]], honor_no_touch: bool = True,
                skip_dynamic: bool = False, triggers: bool = True) -> Level:
    """triggers=True simulates move/toggle/spawn/stop triggers; otherwise skip_dynamic drops gameplay objects
    whose groups any trigger drives (lenient), and without either they stay at their static positions."""
    table = load_objects()
    moving = dynamic_groups(objects, ENGINE_TRIGGER_IDS if triggers else DYNAMIC_TRIGGER_IDS)
    # Objects driven by triggers the engine does not simulate are dropped (lenient) and counted.
    unsimulated = dynamic_groups(objects, APPROX_TRIGGER_IDS) if triggers else set()
    phys = []
    skipped = 0
    for source, obj in enumerate(objects):
        try:
            info = table.get(int(obj.get(KEY_ID, "0")))
        except ValueError:
            continue
        if info is None or info.category not in COLLIDABLE:
            continue
        if honor_no_touch and obj.get(KEY_NO_TOUCH) == "1":
            continue                  # no touch turns any object into decoration (spec §15.2)
        po = make_phys_object(len(phys), obj, info)
        po.source = source
        po.dynamic = bool(moving.intersection(po.groups))
        if unsimulated.intersection(po.groups):
            skipped += 1
            continue
        if po.dynamic and skip_dynamic and not triggers:
            skipped += 1
            continue
        po.index = len(phys)
        phys.append(po)

    def header_int(key: str, default: int = 0) -> int:
        try:
            return int(header.get(key, default))
        except ValueError:
            return default

    mode = header_int("kA2")
    speed = header_int("kA4")
    level = Level(
        objects=phys,
        start_mode=HEADER_GAMEMODES[mode] if 0 <= mode < len(HEADER_GAMEMODES) else "cube",
        start_mini=header_int("kA3") == 1,
        start_speed=HEADER_SPEEDS[speed] if 0 <= speed < len(HEADER_SPEEDS) else 0.9,
        start_upside_down=header_int("kA11") == 1,
        start_dual=header_int("kA8") == 1,
        dynamic_skipped=skipped,
        fix_gravity_bug=header_int("kA32") == 1,
        fix_radius_collision=header_int("kA39") == 1,
    )
    spawn_group = header_int("kA36")
    if spawn_group > 0:
        # Inferred from in-game traces (5/5 levels): the player spawns at the y of the last object, in x order,
        # of the level's spawn group (header kA36); x stays 0.
        members = [o for o in objects if str(spawn_group) in o.get(KEY_GROUPS, "").split(".")]
        if members:
            last = max(enumerate(members), key=lambda im: (_float(im[1], KEY_X), im[0]))[1]
            level.spawn_y = _float(last, KEY_Y) + LEVEL_Y_OFFSET
    if header_int("kA22") == 1 or header_int("kA37") == 1:
        top = max((_float(o, KEY_Y) + LEVEL_Y_OFFSET for o in objects), default=0.0)
        level.max_gameplay_y = max(1200.0, top) + 390.0
    if triggers:
        from .triggers import TriggerEngine, parse_triggers
        trigger_list, approx = parse_triggers(objects)
        level.dynamic_objects = [o for o in phys if o.dynamic]
        level.engine = TriggerEngine(trigger_list, level.dynamic_objects, moving, approx)
    level.build_index()
    return level
