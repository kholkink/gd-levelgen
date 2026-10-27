"""Draws a level with the game's own sprites (read from the local Geometry Dash install) for previews.

The image shows the level as it looks at the start, before any trigger runs: background and ground in their
channel colors, then every visible object's main sprite, transformed (scale, flip, rotation) and tinted with its
main color channel (key 21, else the object's default channel; HSV key 43; additive for blending channels), in
draw order (z layer, z order, object order). Long levels are cut into rows stacked top to bottom.

Approximations: no detail / glow child sprites, animations or particles; channel 0 keeps the sprite's own colors;
triggers, hidden (key 135) and text objects are not drawn.
"""
from __future__ import annotations

import colorsys
import json
import math
import os
import plistlib
import re
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from gdgen.objects import DATA_FILE, load_objects

# Your own Geometry Dash install (Steam default under WSL); override with GD_RESOURCES.
GAME_RESOURCES = Path(os.environ.get("GD_RESOURCES",
                                     "/mnt/c/Program Files (x86)/Steam/steamapps/common/Geometry Dash/Resources"))
SHEETS = ("GJ_GameSheet", "GJ_GameSheet02", "GJ_GameSheet03", "GJ_GameSheet04", "GJ_GameSheetGlow", "PixelSheet_01",
          "FireSheet_01", "GJ_ParticleSheet", "DungeonSheet", "SecretSheet")
HD = 2.0                                   # -hd sheets: 2 px per game unit
LEVEL_Y_OFFSET = 90.0                      # level strings store y relative to the floor top (y = 90)
DEFAULT_COLORS = {1000: (40, 125, 255), 1001: (0, 102, 255), 1002: (255, 255, 255), 1003: (255, 255, 255),
                  1004: (255, 255, 255), 1005: (125, 255, 0), 1006: (0, 255, 255), 1007: (255, 255, 255),
                  1009: (0, 102, 255), 1010: (0, 0, 0), 1011: (255, 255, 255), 1012: (255, 255, 255)}
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _nums(text: str) -> list[float]:
    return [float(v) for v in _NUM.findall(text)]


def _f(obj: dict, key: str, default: float) -> float:
    try:
        return float(obj.get(key, default))
    except ValueError:
        return default


class Atlas:
    """Frame name -> cropped, unrotated RGBA sprite from the game's -hd sprite sheets."""

    def __init__(self, resources: Path = GAME_RESOURCES):
        self.resources = resources
        self.frames: dict[str, tuple[str, dict]] = {}
        for sheet in SHEETS:
            plist = resources / f"{sheet}-hd.plist"
            if plist.exists():
                for name, info in plistlib.loads(plist.read_bytes())["frames"].items():
                    self.frames.setdefault(name, (sheet, info))
        self._sheets: dict[str, Image.Image] = {}

    def _sheet(self, sheet: str) -> Image.Image:
        if sheet not in self._sheets:
            self._sheets[sheet] = Image.open(self.resources / f"{sheet}-hd.png").convert("RGBA")
        return self._sheets[sheet]

    @lru_cache(maxsize=4096)
    def sprite(self, name: str) -> tuple[Image.Image, float, float] | None:
        """(image, offset x, offset y) in -hd pixels; the offset moves the trimmed sprite's centre (y up)."""
        found = self.frames.get(name)
        if found is None:
            return None
        sheet, info = found
        x, y, w, h = _nums(info["textureRect"])
        rotated = info.get("textureRotated", False)
        box = (int(x), int(y), int(x + (h if rotated else w)), int(y + (w if rotated else h)))
        img = self._sheet(sheet).crop(box)
        if rotated:
            img = img.rotate(90, expand=True)
        ox, oy = _nums(info.get("spriteOffset", "{0,0}"))
        return img, ox, oy


def apply_hsv(rgb: tuple[int, int, int], hsv: str) -> tuple[int, int, int]:
    """GD HSV adjustment 'h a s a v a sAdd a vAdd' (hue shift in degrees, saturation/value multiply or add)."""
    parts = hsv.split("a") if hsv else []
    if len(parts) < 5:
        return rgb
    try:
        dh, ds, dv, s_add, v_add = float(parts[0]), float(parts[1]), float(parts[2]), parts[3] == "1", parts[4] == "1"
    except ValueError:
        return rgb
    h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
    h = (h + dh / 360.0) % 1.0
    s = min(1.0, max(0.0, s + ds if s_add else s * ds))
    v = min(1.0, max(0.0, v + dv if v_add else v * dv))
    return tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(h, s, v))


def parse_colors(header: dict[str, str]) -> dict[int, dict]:
    """kS38 color channels -> {channel: {"rgb", "opacity", "blending"}}; player and copied colors resolved."""
    out = {ch: {"rgb": rgb, "opacity": 1.0, "blending": False} for ch, rgb in DEFAULT_COLORS.items()}
    raw = []
    for part in header.get("kS38", "").split("|"):
        kv = part.split("_")
        d = dict(zip(kv[::2], kv[1::2]))
        if "6" in d:
            raw.append(d)
    for d in raw:
        def f(k, default=0.0, d=d):
            try:
                return float(d.get(k, default))
            except ValueError:
                return default
        rgb = (int(f("1")), int(f("2")), int(f("3")))
        player = int(f("4", -1))
        if player in (1, 2):
            rgb = out[1004 + player]["rgb"]
        out[int(f("6"))] = {"rgb": rgb, "opacity": f("7", 1.0), "blending": d.get("5") == "1", "copy": int(f("9")),
                            "hsv": d.get("10", "")}
    for c in out.values():                                       # copied colors, one level deep
        src = c.get("copy")
        if src and src in out:
            c["rgb"] = apply_hsv(out[src]["rgb"], c.get("hsv", ""))
    return out


LEGACY_COLOR_TRIGGERS = {29: 1000, 30: 1001, 104: 1002, 105: 1004, 221: 1, 717: 2, 718: 3, 743: 4, 744: 1003,
                         900: 1009}
COLOR_TRIGGER, ALPHA_TRIGGER, TOGGLE_TRIGGER = 899, 1007, 1049
VIEW_LEAD = 150.0                          # an object is judged by the trigger state when the camera reaches it


class VisualState:
    """Channel colors, group opacities and enabled groups, advanced by color / alpha / toggle triggers in x order.

    Triggers fire instantly when the player's x passes them (fades are taken as finished; spawn-triggered ones
    are skipped)."""

    def __init__(self, header: dict, objects: list[dict]):
        self.colors = parse_colors(header)
        self.alpha: dict[int, float] = {}
        self.disabled: set[int] = set()
        self.events = []
        for o in objects:
            try:
                oid = int(o.get("1", "0"))
            except ValueError:
                continue
            if oid in (COLOR_TRIGGER, ALPHA_TRIGGER, TOGGLE_TRIGGER) or oid in LEGACY_COLOR_TRIGGERS:
                if o.get("62") != "1":
                    self.events.append((_f(o, "2", 0.0), oid, o))
        self.events.sort(key=lambda e: e[0])
        self.next = 0

    def advance(self, x: float) -> None:
        while self.next < len(self.events) and self.events[self.next][0] <= x:
            _, oid, o = self.events[self.next]
            self.next += 1
            if oid == ALPHA_TRIGGER:
                self.alpha[int(_f(o, "51", 0))] = min(1.0, max(0.0, _f(o, "35", 1.0)))
            elif oid == TOGGLE_TRIGGER:
                g = int(_f(o, "51", 0))
                (self.disabled.discard if o.get("56") == "1" else self.disabled.add)(g)
            else:
                ch = int(_f(o, "23", 0)) if oid == COLOR_TRIGGER else LEGACY_COLOR_TRIGGERS[oid]
                if not ch:
                    continue
                rgb = (int(_f(o, "7", 255)), int(_f(o, "8", 255)), int(_f(o, "9", 255)))
                if o.get("15") == "1":
                    rgb = self.colors[1005]["rgb"]
                elif o.get("16") == "1":
                    rgb = self.colors[1006]["rgb"]
                copy = int(_f(o, "50", 0))
                if copy and copy in self.colors:
                    rgb = apply_hsv(self.colors[copy]["rgb"], o.get("49", ""))
                self.colors[ch] = {"rgb": rgb, "opacity": min(1.0, max(0.0, _f(o, "35", 1.0))),
                                   "blending": o.get("17") == "1"}

    def object_look(self, o: dict, default_channel: int) -> tuple[tuple | None, float, bool, bool]:
        """(tint or None, opacity, additive, visible) of an object under the current state."""
        groups = [int(g) for g in o.get("57", "").split(".") if g.isdigit()]
        if any(g in self.disabled for g in groups):
            return None, 0.0, False, False
        alpha = 1.0
        for g in groups:
            alpha *= self.alpha.get(g, 1.0)
        channel = int(_f(o, "21", 0)) or default_channel
        if not channel:
            return None, alpha, False, alpha > 0.02
        c = self.colors.get(channel) or {"rgb": (255, 255, 255), "opacity": 1.0, "blending": False}
        tint = apply_hsv(c["rgb"], o.get("43", "")) if o.get("41") == "1" else c["rgb"]
        alpha *= c["opacity"]
        return tint, alpha, c["blending"], alpha > 0.02


@lru_cache(maxsize=1)
def default_channels() -> dict[int, int]:
    """Object id -> its default main color channel (0: the sprite keeps its own colors)."""
    raw = json.loads(Path(DATA_FILE).read_text())["objects"]
    return {o["id"]: int(o["default_color"]["base"][1]) for o in raw.values()}


class Renderer:
    def __init__(self, resources: Path = GAME_RESOURCES):
        self.atlas = Atlas(resources)
        self.table = load_objects()
        self.defaults = default_channels()
        self.missing: dict[str, int] = {}

    @lru_cache(maxsize=8192)
    def _transformed(self, frame: str, sx: float, sy: float, rot: float, tint: tuple | None, alpha: float,
                     scale: float):
        """The sprite scaled to output pixels, flipped, tinted and rotated (clockwise degrees)."""
        found = self.atlas.sprite(frame)
        if found is None:
            return None
        img, ox, oy = found
        k = scale / HD
        w, h = max(1, int(round(img.width * abs(sx) * k))), max(1, int(round(img.height * abs(sy) * k)))
        img = img.resize((w, h), Image.BILINEAR)
        if sx < 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if sy < 0:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if tint is not None:
            r, g, b, a = img.split()
            r, g, b = (ch.point(lambda v, t=t: v * t // 255) for ch, t in zip((r, g, b), tint))
            img = Image.merge("RGBA", (r, g, b, a))
        if alpha < 1.0:
            img.putalpha(img.getchannel("A").point(lambda v, m=alpha: int(v * m)))
        if rot % 360:
            img = img.rotate(-rot, resample=Image.BILINEAR, expand=True)
        # where the trimmed sprite's centre lies, in output pixels after the same transform (y down)
        dx, dy = ox * sx * k, -oy * sy * k
        a = math.radians(rot)
        return img, dx * math.cos(a) - dy * math.sin(a), dx * math.sin(a) + dy * math.cos(a)

    def render(self, header: dict, objects: list[dict], scale: float = 0.5, row_units: float = 2400.0,
               row_height: float = 720.0, max_rows: int = 40) -> Image.Image:
        """Rows of row_units x row_height game units; each row's window follows the gameplay in it vertically."""
        drawable = []
        for i, o in enumerate(objects):
            try:
                oid = int(o.get("1", "0"))
            except ValueError:
                continue
            info = self.table.get(oid)
            if info is None or info.category == "trigger" or o.get("135") == "1" or not info.frame:
                continue
            drawable.append([int(_f(o, "24", 1)), int(_f(o, "25", 0)), i, o, info, None])
        state = VisualState(header, objects)
        if not drawable:
            return Image.new("RGB", (64, 64), state.colors[1000]["rgb"])
        xs = [_f(d[3], "2", 0.0) for d in drawable]
        x_lo, x_hi = min(0.0, min(xs)), max(xs) + 60
        rows = min(max_rows, max(1, math.ceil((x_hi - x_lo) / row_units)))
        windows = self._row_windows(drawable, xs, x_lo, rows, row_units, row_height)
        row_h, row_w = int(row_height * scale), int(row_units * scale)
        canvas = Image.new("RGB", (row_w, row_h * rows), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        # one pass in x order: every object's look when the camera reaches it, and each row's background
        order = sorted(range(len(drawable)), key=lambda k: xs[k])
        row_marks = [x_lo + (r + 0.5) * row_units for r in range(rows)]
        r_next = 0
        for k in order:
            x = xs[k]
            while r_next < rows and row_marks[r_next] <= x - VIEW_LEAD:
                state.advance(row_marks[r_next])
                self._row_background(draw, state, r_next, row_w, row_h, windows[r_next][1], scale)
                r_next += 1
            state.advance(x - VIEW_LEAD)
            d = drawable[k]
            d[5] = state.object_look(d[3], self.defaults.get(d[4].id, 0))
        for r in range(r_next, rows):
            state.advance(row_marks[r])
            self._row_background(draw, state, r, row_w, row_h, windows[r][1], scale)

        drawable.sort(key=lambda t: (t[0], t[1], t[2]))
        for _, _, _, o, info, look in drawable:
            tint, alpha, blending, visible = look
            if not visible:
                continue
            x, y = _f(o, "2", 0.0), _f(o, "3", 0.0) + LEVEL_Y_OFFSET
            r = int((x - x_lo) // row_units)
            if r >= rows:
                continue
            y_lo, y_hi = windows[r]
            if y < y_lo - 300 or y > y_hi + 300:
                continue
            uniform = _f(o, "32", 1.0) or 1.0
            sx, sy = _f(o, "128", uniform) or uniform, _f(o, "129", uniform) or uniform
            if o.get("4") == "1":
                sx = -sx
            if o.get("5") == "1":
                sy = -sy
            rot = _f(o, "6", _f(o, "131", 0.0)) % 360.0
            out = self._transformed(info.frame, round(sx, 3), round(sy, 3), round(rot, 1), tint, round(alpha, 2),
                                    scale)
            if out is None:
                self.missing[info.frame] = self.missing.get(info.frame, 0) + 1
                continue
            img, dx, dy = out
            cx = (x - x_lo - r * row_units) * scale + dx
            cy = r * row_h + (y_hi - y) * scale + dy
            left, top = int(round(cx - img.width / 2)), int(round(cy - img.height / 2))
            # keep a sprite inside its own row (rows are separate strips of the level)
            if top >= (r + 1) * row_h or top + img.height <= r * row_h:
                continue
            if blending:
                self._add(canvas, img, left, top)
            else:
                canvas.paste(img, (left, top), img)
        return canvas

    @staticmethod
    def _row_windows(drawable, xs, x_lo, rows, row_units, row_height) -> list[tuple[float, float]]:
        """(y_lo, y_hi) per row: centred on the median height of the row's gameplay, never below the floor area."""
        per_row: list[list[float]] = [[] for _ in range(rows)]
        for d, x in zip(drawable, xs):
            r = int((x - x_lo) // row_units)
            if r < rows and d[4].is_gameplay:
                per_row[r].append(_f(d[3], "3", 0.0) + LEVEL_Y_OFFSET)
        out, last = [], None
        for ys in per_row:
            if ys:
                ys.sort()
                lo = max(0.0, ys[len(ys) // 2] - row_height / 2)
                last = (lo, lo + row_height)
            out.append(last)
        first = next((w for w in out if w), (0.0, row_height))
        return [w or first for w in out]

    @staticmethod
    def _row_background(draw: ImageDraw.ImageDraw, state: VisualState, r: int, row_w: int, row_h: int,
                        y_hi: float, scale: float) -> None:
        top = r * row_h
        gy = min(top + row_h, top + int((y_hi - LEVEL_Y_OFFSET) * scale))
        draw.rectangle([0, top, row_w, gy], fill=state.colors[1000]["rgb"])
        if gy < top + row_h:
            draw.rectangle([0, gy, row_w, top + row_h], fill=state.colors[1001]["rgb"])
            draw.line([0, gy, row_w, gy], fill=state.colors[1002]["rgb"], width=max(1, int(scale * 2)))
        draw.line([0, top, row_w, top], fill=(0, 0, 0), width=2)

    @staticmethod
    def _add(canvas: Image.Image, img: Image.Image, left: int, top: int) -> None:
        """Additive blending (GD 'blending' channels) of img's alpha-weighted color onto the canvas."""
        box = (max(0, left), max(0, top), min(canvas.width, left + img.width), min(canvas.height, top + img.height))
        if box[0] >= box[2] or box[1] >= box[3]:
            return
        part = img.crop((box[0] - left, box[1] - top, box[2] - left, box[3] - top))
        glow = Image.composite(part.convert("RGB"), Image.new("RGB", part.size, (0, 0, 0)), part.getchannel("A"))
        canvas.paste(ImageChops.add(canvas.crop(box), glow), box[:2])
