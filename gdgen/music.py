"""Music conditioning: the song's features over each 300-unit segment of a level, as tokens after SEG:<s>.

A level's x maps to time through its speed portals (the player's x speed only changes there), and the song time
is the level time plus the header's song offset (kA13). The clock of segment s only knows the portals before it
(x < 300 s): that is what a model has placed when it samples SEG:s, so MusicInjector reproduces the training tokens
exactly while generating. For every segment the song features of scripts/song_features.py over its time span become:
    MU:e=<0..7>   loudness (mean rms, dB below the song's peak, 7.5 dB per step; 7 = loudest)
    MU:o=<0..7>   onset density (mean normalized onset strength)
    MU:l=<0..7>   bass (mean < 200 Hz energy, dB below its peak)
    MU:b=<0..9>   one per beat inside the segment: the beat's 30-unit cell
"""
from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np

from gdgen.objects import SPEED_PORTAL_IDS
from gdgen.tokens import SEG_Q, SEG_RANGE, Q, Tokenizer

SEG_UNITS = SEG_Q / Q
HEADER_SPEEDS = [0.9, 0.7, 1.1, 1.3, 1.6]                 # kA4 -> m_playerSpeed
PORTAL_SPEED = {0.5: 0.7, 1.0: 0.9, 2.0: 1.1, 3.0: 1.3, 4.0: 1.6}
X_MUL = {0.7: 5.980002, 0.9: 5.7700018882751465, 1.1: 5.870002, 1.3: 6.000002, 1.6: 6.000002}
LEVELS, BEAT_CELLS = 8, 10                                # see tokens.MUSIC_TOKENS


def x_speed(speed: float) -> float:
    """Units per second at a given m_playerSpeed (240 steps of speed * xMul * 0.25)."""
    return speed * X_MUL[speed] * 0.25 * 240


def level_portals(objects: list[dict]) -> list[tuple[float, float]]:
    """(x, m_playerSpeed) of the level's speed portals in x order (no-touch portals do nothing)."""
    portals = []
    for o in objects:
        try:
            oid = int(o.get("1", "0"))
            if oid in SPEED_PORTAL_IDS and o.get("121") != "1":
                portals.append((float(o.get("2", 0) or 0), PORTAL_SPEED[SPEED_PORTAL_IDS[oid]]))
        except (ValueError, KeyError):
            continue
    return sorted(portals)


class LevelClock:
    """Level x <-> seconds since the start, from the start speed and the speed portals (of objects, or given)."""

    def __init__(self, header: dict, objects: list[dict] = (), portals: list[tuple[float, float]] | None = None):
        try:
            start = HEADER_SPEEDS[int(header.get("kA4", "0") or 0)]
        except (ValueError, IndexError):
            start = 0.9
        self.xs, self.ts, self.speeds = [0.0], [0.0], [start]
        for x, speed in level_portals(objects) if portals is None else portals:
            if x <= 0:                                  # at or behind the start: the level starts at that speed
                self.speeds[0] = speed
                continue
            self.ts.append(self.ts[-1] + (x - self.xs[-1]) / x_speed(self.speeds[-1]))
            self.xs.append(x)
            self.speeds.append(speed)

    def time(self, x: float) -> float:
        i = max(0, bisect.bisect_right(self.xs, x) - 1)
        return self.ts[i] + (x - self.xs[i]) / x_speed(self.speeds[i])

    def x(self, t: float) -> float:
        i = max(0, bisect.bisect_right(self.ts, t) - 1)
        return self.xs[i] + (t - self.ts[i]) * x_speed(self.speeds[i])


def load_features(path: str | Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def _level(value: float, lo: float, hi: float) -> int:
    return int(min(LEVELS - 1, max(0, (value - lo) / (hi - lo) * LEVELS)))


def segment_tokens(features: dict, clock: LevelClock, offset: float, seg: int) -> list[str]:
    """Music tokens of segment seg ([] when the song does not cover it)."""
    x0, x1 = seg * SEG_UNITS, (seg + 1) * SEG_UNITS
    if x1 <= 0:
        return []
    t0, t1 = clock.time(max(x0, 0.0)) + offset, clock.time(x1) + offset
    fps = float(features["fps"])
    n = len(features["rms"])
    f0 = int(t0 * fps)
    f1 = max(f0 + 1, int(t1 * fps))
    if f0 >= n or f1 <= 0:
        return []
    f0, f1 = max(0, f0), min(n, f1)
    rms = float(np.mean(features["rms"][f0:f1].astype(np.float32)))
    onset = float(np.mean(features["onset"][f0:f1].astype(np.float32)))
    low = float(np.mean(features["low"][f0:f1].astype(np.float32)))
    out = [f"MU:e={_level(rms, -60.0, 0.0)}", f"MU:o={_level(onset, 0.0, 1.0)}", f"MU:l={_level(low, -60.0, 0.0)}"]
    beats = features["beats"]
    for b in beats[(beats >= t0) & (beats < t1)]:
        cell = int((clock.x(float(b) - offset) - x0) // (SEG_UNITS / BEAT_CELLS))
        out.append(f"MU:b={min(BEAT_CELLS - 1, max(0, cell))}")
    return out


def level_music(header: dict, objects: list[dict], features: dict, segments: range | None = None) -> dict[int, list[str]]:
    """{segment: music tokens} for the level's segments (or the given range) that the song covers."""
    portals = level_portals(objects)
    portal_xs = [x for x, _ in portals]
    clock, known = None, -1
    try:
        offset = float(header.get("kA13", "0") or 0)
    except ValueError:
        offset = 0.0
    if segments is None:
        xs = [float(o.get("2", 0) or 0) for o in objects if o.get("2")]
        segments = range(0, int(max(xs, default=0.0) // SEG_UNITS) + 1)
    out = {}
    for s in segments:
        n = bisect.bisect_left(portal_xs, s * SEG_UNITS)          # the portals before the segment
        if n != known:
            clock, known = LevelClock(header, portals=portals[:n]), n
        toks = segment_tokens(features, clock, offset, s)
        if toks:
            out[s] = toks
    return out


class MusicInjector:
    """`inject` for GPT.generate: the song's music tokens right after every SEG the model samples.

    It sees every sampled token, decodes each finished segment and keeps the speed portals placed so far, which are
    exactly the portals before the new segment, i.e. the clock level_music computes the training tokens with."""

    def __init__(self, tok: Tokenizer, features: dict, header: dict, offset: float):
        self.tok, self.features, self.header, self.offset = tok, features, header, offset
        self.seg_lo = tok.vocab.index[f"SEG:{SEG_RANGE[0]}"]
        self.portals: list[tuple[float, float]] = []
        self.chunk: list[int] = []

    def seg_of(self, t: int) -> int | None:
        s = t - self.seg_lo
        return s + SEG_RANGE[0] if 0 <= s < SEG_RANGE[1] - SEG_RANGE[0] else None

    def state(self) -> tuple:
        return list(self.portals), list(self.chunk)

    def restore(self, state: tuple) -> None:
        self.portals, self.chunk = list(state[0]), list(state[1])

    def music(self, seg: int) -> list[int]:
        clock = LevelClock(self.header, portals=self.portals)
        return [self.tok.vocab.index[t] for t in segment_tokens(self.features, clock, self.offset, seg)]

    def __call__(self, toks: list[int]) -> list[int]:
        t = toks[-1]
        seg = self.seg_of(t)
        if seg is None:
            self.chunk.append(t)
            return []
        if self.chunk:
            _, objs = self.tok.decode_level(["<OBJECTS>", *self.tok.vocab.detokenize(self.chunk), "<EOS>"],
                                            strict=False)
            self.portals = sorted(self.portals + level_portals(objs))
        self.chunk = [t]
        return self.music(seg)
