"""Level <-> token sequence.

A level becomes:

    <BOS> <HEADER> (H:<key> value)* <COLORS> (<CHAN> (CP:<prop> value)*)* <OBJECTS>
    (SEG:<s> | O:<id> X:<q> Y:<c> YF:<f> attribute*)* <EOS>

Objects are sorted by position. SEG:<s> opens the 300-unit (10 block) segment s and is emitted only when the
segment changes, so any segment boundary is a valid place to start a training window. Inside a segment X is
the offset in quarter units (0..1199); Y is the 120-quarter-unit cell (one block) plus the offset in it.

Attributes of an object, in this order:
    flags   FS:<k.k.k> for a frequent set of "1" flags, else F:<k> each; F0:<k> for an explicit "0"
    groups  G:<id> per group (key 57)
    others  in increasing key order (equal scale X/Y count as the pseudo key "s"):
            A:<k>=<v> for a frequent key/value pair, otherwise by the key's kind (fitted from data):
              hsv     K:<k> HH:<hue> HS:<mode>:<s*100> HV:<mode>:<v*100>
              list    K:<k> value* LEND            ('.'-separated, e.g. spawn remaps)
              astr    K:<k> value* LEND            ('a'-separated, e.g. particle settings)
              text    K:<k> B:<byte>* LEND         (base64 text, as decoded bytes)
              num     K:<k> value
A value is V:<normalized> when it is in the vocabulary, D:<c>* DEND for other numbers, and TRAW B:<byte>* LEND
for anything else.

Lossy only where it is invisible: positions are rounded to 0.25 units, angles to 0.1 degree (mod 360), scales to
0.005 (coarser above 4), other floats to 3 decimals, HSV saturation and brightness to 0.01, and editor-only keys
(and keys never seen when fitting the vocabulary) are dropped. `canonical_object` / `canonical_header` apply
exactly these losses, so decode(encode(x)) == canonical(x) is the round-trip contract.
"""
from __future__ import annotations

import base64
import binascii
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from gdgen.objects import load_objects

VOCAB_FILE = Path(__file__).parent / "data" / "vocab_v1.json"

Q = 4                                  # position quantum: 1/4 unit
SEG_Q = 300 * Q                        # segment width in quanta (10 blocks)
SEG_RANGE = (-8, 2040)                 # segments; x from -2400 to 612000 units
Y_CELL = 30 * Q                        # one block
Y_RANGE = (-64, 448)                   # cells; y from -1920 to 13440 units
INT_RANGE = (-1024, 10000)             # every integer in this range has its own V token
GROUP_MAX = 9999

DROP_KEYS = frozenset({"18", "20", "61", "108", "155", "156"})   # copy id, editor layers, linked group, color-key caches
GROUPS_KEY = "57"
TEXT_KEYS = frozenset({"31"})
HSV_KEYS = frozenset({"43", "44", "49"})
LIST_KEYS = frozenset({"57", "430", "435", "442"})
ANGLE_KEYS = frozenset({"6", "131", "132"})                     # rotation, skew X / skew Y
SCALE_KEYS = frozenset({"32", "128", "129"})                    # legacy uniform scale, scale X, scale Y
SCALE_X, SCALE_Y, UNIFORM_SCALE = "128", "129", "s"
DROP_HEADER = frozenset({"kA14"})                               # editor song guidelines
COLORS_KEY = "kS38"
COLOR_HSV_PROP = "10"

SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<HEADER>", "<COLORS>", "<CHAN>", "<OBJECTS>", "LEND", "DEND", "EMPTY",
           "TRAW", "<UNK>", "O?"]
# Training-stream structure: which object stream follows (<GAMEPLAY> opens the gameplay context of a <DECO> window),
# whether the window starts mid-level (<CONT>), and level metadata (stars 1..10, demon tier 0..6, rating 0 featured
# .. 3 mythic, length 0 tiny .. 4 XL) as conditioning.
STREAM_TOKENS = ["<FULL>", "<LAYOUT>", "<DECO>", "<GAMEPLAY>", "<CONT>"]
META_TOKENS = ([f"M:stars={i}" for i in range(11)] + [f"M:demon={i}" for i in range(7)]
               + [f"M:epic={i}" for i in range(4)] + [f"M:len={i}" for i in range(6)])
RESERVED = 512                         # <R0>.. kept free so later conditioning (audio, style) keeps token ids stable
# Music conditioning (gdgen/music.py) after SEG:<s>; it takes the first reserved slots, so other ids do not move.
MUSIC_TOKENS = ([f"MU:e={i}" for i in range(8)] + [f"MU:o={i}" for i in range(8)] + [f"MU:l={i}" for i in range(8)]
                + [f"MU:b={i}" for i in range(10)])
DIGIT_CHARS = "0123456789.-+eE"
ATTR_PREFIXES = ("FS:", "F:", "F0:", "G:", "A:", "K:")
_HSV_RE = re.compile(r"^-?[\d.]+a-?[\d.]+a-?[\d.]+a[01]a[01]$")
_HEADER_KEY_RE = re.compile(r"^k([AS])(\d{1,2})$")


def norm_num(v: str) -> str | None:
    """Canonical text of a number: integers without a point, others rounded to 3 decimals. None if not numeric."""
    try:
        f = float(v)
    except ValueError:
        return None
    if not math.isfinite(f):
        return None
    if abs(f - round(f)) < 1e-9 and abs(f) < 1e12:
        return str(int(round(f)))
    s = f"{f:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def norm_angle(v: str) -> str | None:
    n = norm_num(v)
    if n is None:
        return None
    return norm_num(str((round(float(n) * 10) % 3600) / 10))


def norm_scale(v: str) -> str | None:
    n = norm_num(v)
    if n is None:
        return None
    f = float(n)
    step = 0.005 if abs(f) < 4 else 0.05 if abs(f) < 20 else 0.5
    return norm_num(str(round(f / step) * step))


def norm_key_num(k: str, v: str) -> str | None:
    """norm_num with the key-specific quantization of angles and scales."""
    if k in ANGLE_KEYS:
        return norm_angle(v)
    if k in SCALE_KEYS:
        return norm_scale(v)
    return norm_num(v)


def quant(v: float) -> int:
    return int(math.floor(v * Q + 0.5))


def kept_key(k: str) -> bool:
    """Attribute keys the tokenizer keeps: numeric 4..1023 minus editor-only ones (start-pos kA* keys are dropped)."""
    return k.isdigit() and 3 < int(k) < 1024 and k not in DROP_KEYS


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


@dataclass
class HSV:
    hue: int
    s_add: int
    s: int          # x100
    v_add: int
    v: int          # x100

    @classmethod
    def parse(cls, text: str) -> "HSV | None":
        if not _HSV_RE.match(text):
            return None
        h, s, v, sa, va = text.split("a")
        try:
            s_add, v_add, h, s, v = int(sa), int(va), float(h), float(s), float(v)
        except ValueError:
            return None
        s_lo, s_hi = (-100, 100) if s_add else (0, 200)
        v_lo, v_hi = (-100, 100) if v_add else (0, 200)
        return cls(hue=clamp(int(round(h)), -180, 180),
                   s_add=s_add, s=clamp(int(round(s * 100)), s_lo, s_hi),
                   v_add=v_add, v=clamp(int(round(v * 100)), v_lo, v_hi))

    def text(self) -> str:
        return f"{self.hue}a{norm_num(str(self.s / 100))}a{norm_num(str(self.v / 100))}a{self.s_add}a{self.v_add}"

    def tokens(self) -> list[str]:
        return [f"HH:{self.hue}", f"HS:{self.s_add}:{self.s}", f"HV:{self.v_add}:{self.v}"]


@dataclass
class Vocab:
    tokens: list[str]
    key_kinds: dict[str, str]                     # object key -> flag|groups|hsv|list|astr|text|num
    index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self.index = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: str) -> bool:
        return token in self.index

    def ids(self, tokens: list[str]) -> list[int]:
        unk = self.index["<UNK>"]
        return [self.index.get(t, unk) for t in tokens]

    def detokenize(self, ids) -> list[str]:
        return [self.tokens[i] for i in ids]

    def save(self, path: str | Path = VOCAB_FILE) -> None:
        Path(path).write_text(json.dumps({"version": 1, "tokens": self.tokens, "key_kinds": self.key_kinds}))

    @classmethod
    def load(cls, path: str | Path = VOCAB_FILE) -> "Vocab":
        raw = json.loads(Path(path).read_text())
        return cls(raw["tokens"], raw["key_kinds"])


# ----------------------------------------------------------------------------------------------- vocabulary fit


class KindCounter:
    """Fit pass 1: per object key, the evidence for its kind; plus the non-integer values worth a V token."""

    def __init__(self):
        self.seen: Counter = Counter()
        self.non_flag: set[str] = set()
        self.astr: Counter = Counter()
        self.hsv_ok: Counter = Counter()
        self.multi_dot: Counter = Counter()
        self.values: Counter = Counter()

    def merge(self, other: "KindCounter") -> None:
        self.seen.update(other.seen)
        self.non_flag |= other.non_flag
        self.astr.update(other.astr)
        self.hsv_ok.update(other.hsv_ok)
        self.multi_dot.update(other.multi_dot)
        self.values.update(other.values)

    def add_value(self, n: str | None) -> None:
        if n is not None and not _is_static_int(n):
            self.values[n] += 1

    def add_object(self, obj: dict[str, str]) -> None:
        for k, v in obj.items():
            if not kept_key(k):
                continue
            self.seen[k] += 1
            if v not in ("0", "1"):
                self.non_flag.add(k)
            if k in TEXT_KEYS:
                continue
            if "a" in v:
                self.astr[k] += 1
                if _HSV_RE.match(v):
                    self.hsv_ok[k] += 1
                elif k not in HSV_KEYS:
                    for part in v.split("a"):
                        self.add_value(norm_num(part))
            elif k in LIST_KEYS or v.count(".") >= 2:
                if v.count(".") >= 2:
                    self.multi_dot[k] += 1
                if k != GROUPS_KEY:
                    for part in v.split("."):
                        self.add_value(norm_num(part))
            else:
                self.add_value(norm_key_num(k, v))

    def add_header(self, header: dict[str, str]) -> None:
        for k, v in header.items():
            if k not in DROP_HEADER and k != COLORS_KEY:
                self.add_value(norm_num(v))
        for chan in _color_channels(header.get(COLORS_KEY, "")):
            for k, v in chan.items():
                if k != COLOR_HSV_PROP:
                    self.add_value(norm_num(v))

    def kinds(self) -> dict[str, str]:
        out = {}
        for k in self.seen:
            if k == GROUPS_KEY:
                out[k] = "groups"
            elif k in TEXT_KEYS:
                out[k] = "text"
            elif k in HSV_KEYS or (self.astr[k] and self.hsv_ok[k] == self.astr[k]):
                out[k] = "hsv"
            elif self.astr[k]:
                out[k] = "astr"
            elif k in LIST_KEYS or self.multi_dot[k]:
                out[k] = "list"
            elif k not in self.non_flag:
                out[k] = "flag"
            else:
                out[k] = "num"
        return out


class PairCounter:
    """Fit pass 2 (needs the kinds): frequent flag sets and key=value pairs, which get single tokens."""

    def __init__(self, kinds: dict[str, str]):
        self.kinds = kinds
        self.pairs: Counter = Counter()
        self.flagsets: Counter = Counter()

    def merge(self, other: "PairCounter") -> None:
        self.pairs.update(other.pairs)
        self.flagsets.update(other.flagsets)

    def add_object(self, obj: dict[str, str]) -> None:
        c = canonical_object(obj, self.kinds)
        if c is None:
            return
        flags, _, _, rest = split_attributes(c, self.kinds)
        if len(flags) > 1:
            self.flagsets[".".join(flags)] += 1
        for k, v, kind in rest:
            if kind in ("num", "hsv"):
                self.pairs[(k, v)] += 1


def _is_static_int(n: str) -> bool:
    try:
        i = int(n)
    except ValueError:
        return False
    return INT_RANGE[0] <= i < INT_RANGE[1]


def build_vocab(kc: KindCounter, pc: PairCounter | None = None, max_float_values: int = 8000,
                max_pairs: int = 22000, max_flagsets: int = 1024, min_count: int = 3) -> Vocab:
    toks = [*SPECIAL, *STREAM_TOKENS, *META_TOKENS, *MUSIC_TOKENS,
            *(f"<R{i}>" for i in range(len(MUSIC_TOKENS), RESERVED))]
    toks += [f"O:{i}" for i in sorted(load_objects())]
    toks += [f"SEG:{s}" for s in range(*SEG_RANGE)]
    toks += [f"X:{q}" for q in range(SEG_Q)]
    toks += [f"Y:{c}" for c in range(*Y_RANGE)]
    toks += [f"YF:{f}" for f in range(Y_CELL)]
    kinds = kc.kinds()
    for k in sorted(kinds, key=int):
        toks += [f"K:{k}", f"F:{k}", f"F0:{k}"] if kinds[k] == "flag" else [f"K:{k}"]
    toks += [f"H:kA{k}" for k in range(1, 100)] + [f"H:kS{k}" for k in range(1, 100)]
    toks += [f"CP:{k}" for k in range(1, 40)]
    toks += [f"G:{g}" for g in range(GROUP_MAX + 1)]
    toks += [f"HH:{h}" for h in range(-180, 181)]
    for mode, (lo, hi) in ((0, (0, 200)), (1, (-100, 100))):
        toks += [f"HS:{mode}:{s}" for s in range(lo, hi + 1)]
        toks += [f"HV:{mode}:{v}" for v in range(lo, hi + 1)]
    toks += [f"B:{b}" for b in range(256)]
    toks += [f"D:{c}" for c in DIGIT_CHARS]
    toks += [f"V:{i}" for i in range(*INT_RANGE)]
    toks += [f"V:{v}" for v, n in kc.values.most_common(max_float_values) if n >= min_count]
    if pc is not None:
        toks += [f"FS:{s}" for s, n in pc.flagsets.most_common(max_flagsets) if n >= min_count]
        toks += [f"A:{k}={v}" for (k, v), n in pc.pairs.most_common(max_pairs) if n >= min_count]
    assert len(toks) <= 65535, f"vocab of {len(toks)} tokens does not fit uint16 shards"
    return Vocab(toks, kinds)


# ----------------------------------------------------------------------------------------------- canonical form


def _color_channels(text: str) -> list[dict[str, str]]:
    chans = []
    for part in text.split("|"):
        if part:
            kv = part.split("_")
            chans.append(dict(zip(kv[::2], kv[1::2])))
    return chans


def _key_order(k: str):
    return (0, int(k), "") if k.isdigit() else (1, 0, k)


def _header_order(k: str):
    m = _HEADER_KEY_RE.match(k)
    return (m.group(1), int(m.group(2)))


def _b64_bytes(v: str) -> bytes | None:
    try:
        raw = base64.urlsafe_b64decode(v + "=" * (-len(v) % 4))
    except (binascii.Error, ValueError):
        return None
    return raw if base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=") == v.rstrip("=") else None


def _groups_ok(v: str) -> bool:
    parts = [g for g in v.split(".") if g]
    return bool(parts) and all(g.isdigit() and int(g) <= GROUP_MAX for g in parts)


def attr_kind(kinds: dict[str, str], k: str, v: str) -> str:
    """The encoding actually used for this value (falls back when a value does not fit its key's kind)."""
    kind = kinds.get(k, "num")
    if kind == "flag" and v not in ("0", "1"):
        return "num"
    if kind == "groups" and not _groups_ok(v):
        return "list"
    if kind == "hsv" and HSV.parse(v) is None:
        return "astr"
    if kind == "text" and _b64_bytes(v) is None:
        return "num"
    return kind


def canonical_value(k: str, kind: str, v: str) -> str:
    if kind == "flag":
        return v
    if kind == "groups":
        return ".".join(str(int(g)) for g in v.split(".") if g)
    if kind == "hsv":
        return HSV.parse(v).text()
    if kind == "text":
        return base64.urlsafe_b64encode(_b64_bytes(v)).decode("ascii")
    if kind == "list":
        return ".".join(norm_num(p) or p for p in v.split("."))
    if kind == "astr":
        return "a".join(norm_num(p) or p for p in v.split("a"))
    return norm_key_num(k, v) or v


def object_position(obj: dict[str, str]) -> tuple[int, int, int, int]:
    """(segment, x in segment, y cell, y in cell), all in quanta and clamped to the token ranges."""
    xq = quant(float(obj.get("2", "0") or 0))
    yq = quant(float(obj.get("3", "0") or 0))
    seg = clamp(xq // SEG_Q, SEG_RANGE[0], SEG_RANGE[1] - 1)
    xin = clamp(xq - seg * SEG_Q, 0, SEG_Q - 1)
    cell = clamp(yq // Y_CELL, Y_RANGE[0], Y_RANGE[1] - 1)
    yf = clamp(yq - cell * Y_CELL, 0, Y_CELL - 1)
    return seg, xin, cell, yf


def position_text(seg: int, xin: int, cell: int, yf: int) -> tuple[str, str]:
    return norm_num(str((seg * SEG_Q + xin) / Q)), norm_num(str((cell * Y_CELL + yf) / Q))


def canonical_object(obj: dict[str, str], kinds: dict[str, str]) -> dict[str, str] | None:
    """The object as the tokenizer preserves it (quantized position, normalized values, no editor keys)."""
    try:
        oid = int(obj["1"])
        pos = object_position(obj)
    except (KeyError, ValueError):
        return None
    x, y = position_text(*pos)
    out = {"1": str(oid), "2": x, "3": y}
    for k in sorted((k for k in obj if kept_key(k) and k in kinds), key=_key_order):
        out[k] = canonical_value(k, attr_kind(kinds, k, obj[k]), obj[k])
    return out


def split_attributes(c: dict[str, str], kinds: dict[str, str]):
    """Canonical object -> (flags set to 1, flags set to 0, groups or None, [(key, value, kind)] in token order)."""
    flags, zeros, groups, rest = [], [], None, []
    for k, v in c.items():
        if k in ("1", "2", "3"):
            continue
        kind = attr_kind(kinds, k, v)
        if kind == "flag":
            (flags if v == "1" else zeros).append(k)
        elif kind == "groups":
            groups = v
        elif k in (SCALE_X, SCALE_Y) and c.get(SCALE_X) == c.get(SCALE_Y):
            if k == SCALE_X:
                rest.append((UNIFORM_SCALE, v, "num"))
        else:
            rest.append((k, v, kind))
    return flags, zeros, groups, rest


def canonical_header(header: dict[str, str]) -> dict[str, str]:
    out = {}
    for k in sorted((k for k in header if _HEADER_KEY_RE.match(k)), key=_header_order):
        if k in DROP_HEADER or k == COLORS_KEY:
            continue
        out[k] = norm_num(header[k]) or header[k]
    if COLORS_KEY in header:
        chans = []
        for chan in _color_channels(header[COLORS_KEY]):
            c = {}
            for k in sorted((k for k in chan if k.isdigit() and 1 <= int(k) < 40), key=_key_order):
                hsv = HSV.parse(chan[k]) if k == COLOR_HSV_PROP else None
                c[k] = hsv.text() if hsv else (norm_num(chan[k]) or chan[k])
            chans.append(c)
        out[COLORS_KEY] = _colors_text(chans)
    return out


def _colors_text(chans: list[dict[str, str]]) -> str:
    return "|".join("_".join(f"{k}_{v}" for k, v in c.items()) for c in chans) + "|"


# ----------------------------------------------------------------------------------------------- tokenizer


class Tokenizer:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.kinds = vocab.key_kinds

    # ---- encoding

    def value(self, v: str) -> list[str]:
        if v == "":
            return ["EMPTY"]
        n = norm_num(v)
        if n is not None:
            if f"V:{n}" in self.vocab:
                return [f"V:{n}"]
            if all(c in DIGIT_CHARS for c in n):
                return [f"D:{c}" for c in n] + ["DEND"]
        return ["TRAW", *(f"B:{b}" for b in v.encode("utf-8")), "LEND"]

    def attribute(self, k: str, v: str, kind: str) -> list[str]:
        pair = f"A:{k}={v}"
        if kind in ("num", "hsv") and pair in self.vocab:
            return [pair]
        if k == UNIFORM_SCALE:
            return [*self.attribute(SCALE_X, v, kind), *self.attribute(SCALE_Y, v, kind)]
        if kind == "hsv":
            return [f"K:{k}", *HSV.parse(v).tokens()]
        if kind == "text":
            return [f"K:{k}", *(f"B:{b}" for b in _b64_bytes(v)), "LEND"]
        if kind in ("list", "astr", "groups"):
            out = [f"K:{k}"]
            for part in v.split("a" if kind == "astr" else "."):
                out += self.value(part)
            return out + ["LEND"]
        return [f"K:{k}", *self.value(v)]

    def encode_object(self, obj: dict[str, str]) -> tuple[tuple[int, int, int, int], list[str]] | None:
        """(position, tokens) of one object; tokens start with its O token. None for objects without an id."""
        c = canonical_object(obj, self.kinds)
        if c is None:
            return None
        pos = object_position(c)
        oid = f"O:{c['1']}"
        out = [oid] if oid in self.vocab else ["O?", *self.value(c["1"])]
        out += [f"X:{pos[1]}", f"Y:{pos[2]}", f"YF:{pos[3]}"]
        flags, zeros, groups, rest = split_attributes(c, self.kinds)
        if len(flags) > 1 and f"FS:{'.'.join(flags)}" in self.vocab:
            out.append(f"FS:{'.'.join(flags)}")
        else:
            out += [f"F:{k}" for k in flags]
        out += [f"F0:{k}" for k in zeros]
        if groups is not None:
            out += [f"G:{g}" for g in groups.split(".")]
        for k, v, kind in rest:
            out += self.attribute(k, v, kind)
        return pos, out

    def encode_header(self, header: dict[str, str]) -> list[str]:
        h = canonical_header(header)
        out = ["<HEADER>"]
        for k, v in h.items():
            if k != COLORS_KEY and f"H:{k}" in self.vocab:
                out += [f"H:{k}", *self.value(v)]
        out.append("<COLORS>")
        for chan in _color_channels(h.get(COLORS_KEY, "")):
            out.append("<CHAN>")
            for k, v in chan.items():
                hsv = HSV.parse(v) if k == COLOR_HSV_PROP else None
                out += [f"CP:{k}", *(hsv.tokens() if hsv else self.value(v))]
        return out

    def encode_objects(self, objects: list[dict[str, str]], music: dict[int, list[str]] | None = None) -> list[str]:
        """Objects sorted by position (x, then y, then original order), with SEG tokens. With music ({segment:
        MU tokens}, gdgen/music.py) every segment that has objects or music gets its SEG, followed by its music."""
        enc = []
        for i, o in enumerate(objects):
            e = self.encode_object(o)
            if e is not None:
                pos, toks = e
                enc.append(((*pos, i), toks))
        enc.sort(key=lambda t: t[0])
        music = music or {}
        pending = sorted(s for s in music if s not in {e[0][0] for e in enc})       # music-only segments
        out, cur, k = [], None, 0
        for (seg, *_), toks in enc:
            if seg != cur:
                while k < len(pending) and pending[k] < seg:
                    out += [f"SEG:{pending[k]}", *music[pending[k]]]
                    k += 1
                out += [f"SEG:{seg}", *music.get(seg, [])]
                cur = seg
            out += toks
        for s in pending[k:]:
            out += [f"SEG:{s}", *music[s]]
        return out

    def encode_level(self, header: dict[str, str], objects: list[dict[str, str]],
                     music: dict[int, list[str]] | None = None) -> list[str]:
        return ["<BOS>", *self.encode_header(header), "<OBJECTS>", *self.encode_objects(objects, music), "<EOS>"]

    # ---- decoding

    def decode_level(self, tokens: list[str], strict: bool = True) -> tuple[dict[str, str], list[dict[str, str]]]:
        """Tokens -> (header, objects). Non-strict decoding (model samples) drops a malformed element and resumes at
        the next element start instead of raising; the number of dropped elements is in self.last_errors."""
        r = _Reader(tokens)
        header: dict[str, str] = {}
        chans: list[dict[str, str]] = []
        objects: list[dict[str, str]] = []
        seg = 0
        self.last_errors = 0
        while not r.done():
            t = r.next()
            if t in ("<BOS>", "<HEADER>", "<COLORS>", "<OBJECTS>", "<PAD>", *STREAM_TOKENS) or t.startswith(("M:", "MU:")):
                continue
            if t == "<EOS>":
                break
            try:
                if t.startswith("H:"):
                    header[t[2:]] = self._read_value(r)
                elif t == "<CHAN>":
                    chans.append({})
                elif t.startswith("CP:"):
                    k = t[3:]
                    chans[-1][k] = self._read_hsv(r).text() if r.peek().startswith("HH:") else self._read_value(r)
                elif t.startswith("SEG:"):
                    seg = int(t[4:])
                elif t.startswith("O:") or t == "O?":
                    objects.append(self._read_object(t, r, seg))
                else:
                    raise ValueError(f"unexpected token {t!r} at {r.i - 1}")
            except (ValueError, IndexError, KeyError, UnicodeDecodeError):
                if strict:
                    raise
                self.last_errors += 1
                while not r.done() and not r.peek().startswith(("O:", "O?", "SEG:", "H:", "CP:", "<CHAN>", "<EOS>")):
                    r.next()
        if chans:
            header[COLORS_KEY] = _colors_text(chans)
        return header, objects

    def _read_object(self, first: str, r: "_Reader", seg: int) -> dict[str, str]:
        oid = first[2:] if first != "O?" else self._read_value(r)
        xin = int(r.expect("X:")[2:])
        cell = int(r.expect("Y:")[2:])
        yf = int(r.expect("YF:")[3:])
        x, y = position_text(seg, xin, cell, yf)
        o = {"1": oid, "2": x, "3": y}
        groups: list[str] = []
        while r.peek().startswith(ATTR_PREFIXES):
            t = r.next()
            if t.startswith("G:"):
                groups.append(t[2:])
            elif t.startswith("FS:"):
                o.update({k: "1" for k in t[3:].split(".")})
            elif t.startswith("F:"):
                o[t[2:]] = "1"
            elif t.startswith("F0:"):
                o[t[3:]] = "0"
            elif t.startswith("A:"):
                k, v = t[2:].split("=", 1)
                if k == UNIFORM_SCALE:
                    o[SCALE_X] = o[SCALE_Y] = v
                else:
                    o[k] = v
            else:
                o[t[2:]] = self._read_attr_value(t[2:], r)
        if groups:
            o[GROUPS_KEY] = ".".join(groups)
        head = {k: o[k] for k in ("1", "2", "3")}
        return {**head, **{k: o[k] for k in sorted((k for k in o if k not in head), key=_key_order)}}

    def _read_attr_value(self, k: str, r: "_Reader") -> str:
        kind = self.kinds.get(k, "num")
        nxt = r.peek()
        if nxt.startswith("HH:"):
            return self._read_hsv(r).text()
        if kind == "text" and (nxt.startswith("B:") or nxt == "LEND"):
            return base64.urlsafe_b64encode(bytes(self._read_bytes(r))).decode("ascii")
        if kind in ("list", "groups", "astr", "hsv"):
            parts = []
            while r.peek() != "LEND":
                parts.append(self._read_value(r))
            r.next()
            return ("a" if kind in ("astr", "hsv") else ".").join(parts)
        return self._read_value(r)

    def _read_bytes(self, r: "_Reader") -> list[int]:
        raw = []
        while (t := r.next()) != "LEND":
            if not t.startswith("B:"):
                raise ValueError(f"expected a byte, got {t!r} at {r.i - 1}")
            raw.append(int(t[2:]))
        return raw

    def _read_hsv(self, r: "_Reader") -> HSV:
        h = int(r.expect("HH:")[3:])
        _, sm, s = r.expect("HS:").split(":")
        _, vm, v = r.expect("HV:").split(":")
        return HSV(hue=h, s_add=int(sm), s=int(s), v_add=int(vm), v=int(v))

    def _read_value(self, r: "_Reader") -> str:
        t = r.next()
        if t.startswith("V:"):
            return t[2:]
        if t == "EMPTY":
            return ""
        if t == "TRAW":
            return bytes(self._read_bytes(r)).decode("utf-8", "replace")
        if t.startswith("D:"):
            chars = [t[2:]]
            while (t := r.next()) != "DEND":
                if not t.startswith("D:"):
                    raise ValueError(f"expected a digit, got {t!r} at {r.i - 1}")
                chars.append(t[2:])
            return "".join(chars)
        raise ValueError(f"expected a value, got {t!r} at {r.i - 1}")


class _Reader:
    def __init__(self, tokens: list[str]):
        self.t = tokens
        self.i = 0

    def done(self) -> bool:
        return self.i >= len(self.t)

    def peek(self) -> str:
        return self.t[self.i] if self.i < len(self.t) else "<EOS>"

    def next(self) -> str:
        t = self.peek()
        self.i += 1
        return t

    def expect(self, prefix: str) -> str:
        t = self.next()
        if not t.startswith(prefix):
            raise ValueError(f"expected {prefix}..., got {t!r} at {self.i - 1}")
        return t
