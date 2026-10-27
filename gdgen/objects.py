"""Object table for GD 2.2081 (extracted from the game by re/extract_objects.py) plus coarse categories.

Categories are what the rest of the pipeline uses:
  solid, slope, hazard, orb, pad, portal_mode, portal_gravity, portal_mirror, portal_size, portal_dual,
  teleport, speed, collectible, start_pos, trigger, deco
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "objects_2.2081.json"

GAMEMODE_BY_TYPE = {
    "CubePortal": "cube", "ShipPortal": "ship", "BallPortal": "ball", "UfoPortal": "ufo", "WavePortal": "wave",
    "RobotPortal": "robot", "SpiderPortal": "spider", "SwingPortal": "swing",
}
_CATEGORY_BY_TYPE = {
    "Solid": "solid", "Breakable": "solid",
    "Slope": "slope",
    "Hazard": "hazard", "AnimatedHazard": "hazard",
    "YellowJumpRing": "orb", "PinkJumpRing": "orb", "GravityRing": "orb", "RedJumpRing": "orb", "GreenRing": "orb",
    "DropRing": "orb", "DashRing": "orb", "GravityDashRing": "orb", "CustomRing": "orb", "SpiderOrb": "orb",
    "TeleportOrb": "orb",
    "YellowJumpPad": "pad", "PinkJumpPad": "pad", "GravityPad": "pad", "RedJumpPad": "pad", "SpiderPad": "pad",
    **{t: "portal_mode" for t in GAMEMODE_BY_TYPE},
    "NormalGravityPortal": "portal_gravity", "InverseGravityPortal": "portal_gravity",
    "GravityTogglePortal": "portal_gravity",
    "NormalMirrorPortal": "portal_mirror", "InverseMirrorPortal": "portal_mirror",
    "RegularSizePortal": "portal_size", "MiniSizePortal": "portal_size",
    "DualPortal": "portal_dual", "SoloPortal": "portal_dual",
    "TeleportPortal": "teleport",
    "SecretCoin": "collectible", "UserCoin": "collectible", "Collectible": "collectible",
    "Decoration": "deco", "Special": "deco", "EnterEffectObject": "trigger", "CollisionObject": "trigger",
}
SPEED_PORTAL_IDS = {200: 0.5, 201: 1.0, 202: 2.0, 203: 3.0, 1334: 4.0}
START_POS_ID = 31
GAMEPLAY_CATEGORIES = frozenset({
    "solid", "slope", "hazard", "orb", "pad", "portal_mode", "portal_gravity", "portal_mirror", "portal_size",
    "portal_dual", "teleport", "speed", "start_pos",
})


@dataclass(frozen=True)
class ObjectInfo:
    id: int
    frame: str
    cls: str | None
    type: str
    category: str
    width: float           # unscaled size used for the hitbox (m_width / m_height)
    height: float
    hitbox_mul: tuple[float, float]
    radius: float | None   # circular hitbox (saws) when set

    @property
    def hitbox(self) -> tuple[float, float]:
        """Unrotated, unscaled hitbox size in game units (one grid cell = 30)."""
        if self.radius:
            return (2 * self.radius, 2 * self.radius)
        return (self.width * self.hitbox_mul[0], self.height * self.hitbox_mul[1])

    @property
    def is_gameplay(self) -> bool:
        return self.category in GAMEPLAY_CATEGORIES


def _category(obj: dict) -> str:
    if obj["id"] == START_POS_ID:
        return "start_pos"
    if obj["id"] in SPEED_PORTAL_IDS:
        return "speed"
    if obj["type"] in ("Hazard", "AnimatedHazard"):      # includes 2.2 invisible damage zones (3610/3611)
        return "hazard"
    if obj["type"] == "Modifier" or (obj["frame"] or "").startswith("edit_e"):
        return "trigger"
    return _CATEGORY_BY_TYPE.get(obj["type"], "deco")


@lru_cache(maxsize=1)
def load_objects(path: str | Path = DATA_FILE) -> dict[int, ObjectInfo]:
    raw = json.loads(Path(path).read_text())["objects"]
    table = {}
    for obj in raw.values():
        sw, sh = obj.get("sprite_size") or (30.0, 30.0)
        ow, oh = obj["size_override"]
        table[obj["id"]] = ObjectInfo(
            id=obj["id"],
            frame=obj["frame"],
            cls=obj["class"],
            type=obj["type"],
            category=_category(obj),
            width=ow if ow is not None and not math.isnan(ow) else sw,
            height=oh if oh is not None and not math.isnan(oh) else sh,
            hitbox_mul=tuple(obj["hitbox_mul"]),
            radius=obj["radius"],
        )
    return table


def category_of(object_id: int) -> str:
    info = load_objects().get(object_id)
    return info.category if info else "unknown"


def is_layout(obj: dict[str, str]) -> bool:
    """Gameplay object of the layout stream: not decoration, triggers, coins, start positions or no-touch objects."""
    try:
        cat = category_of(int(obj.get("1", "0")))
    except ValueError:
        return False
    return cat in GAMEPLAY_CATEGORIES and cat != "start_pos" and obj.get("121") != "1"
