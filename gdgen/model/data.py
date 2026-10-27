"""Training windows cut from the pretokenized shards (scripts/pretokenize.py).

Two window kinds, both `ctx_len + 1` tokens (inputs = [:-1], targets = [1:], -100 where no loss):

    layout  <BOS> <LAYOUT> M:* header <OBJECTS> [<CONT>] layout objects from a segment boundary ...
            A fresh window (start of the level) also trains the header; a <CONT> window only the objects.
    deco    <BOS> <DECO> M:* header <GAMEPLAY> layout objects of segments s-1 .. e+2 <OBJECTS> [<CONT>]
            tail of the deco objects of segment s-1, then the deco objects of segments s .. e (the only loss).
            The model sees the gameplay it decorates, including two segments ahead.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from gdgen.tokens import SEG_RANGE, Vocab

IGNORE = -100


def token_dirs(root: str | Path) -> list[Path]:
    """The pretokenized directories under root: root itself, or its parts (any subdirectory with an index.jsonl)."""
    root = Path(root)
    if (root / "index.jsonl").exists():
        return [root]
    dirs = sorted(p.parent for p in root.glob("*/index.jsonl")) or sorted(p.parent for p in root.glob("**/index.jsonl"))
    if not dirs:
        raise FileNotFoundError(f"no index.jsonl under {root}")
    return dirs


class TokenData:
    """Index + memory-mapped shards of a pretokenized dataset (one directory or several parts with one vocab)."""

    def __init__(self, root: str | Path, split: str = "train", val_every: int = 50, min_game_version: int = 21):
        self.dirs = token_dirs(root)
        self.vocab = Vocab.load(self.dirs[0] / "vocab.json")
        for d in self.dirs[1:]:
            assert (d / "vocab.json").read_text() == (self.dirs[0] / "vocab.json").read_text(), f"vocab differs in {d}"
        keep = (lambda r: r["id"] % val_every == 0) if split == "val" else (lambda r: r["id"] % val_every != 0)
        self.recs = []
        for di, d in enumerate(self.dirs):
            for line in (d / "index.jsonl").read_text().splitlines():
                r = json.loads(line) if line else None
                if r and keep(r) and r["game_version"] >= min_game_version and r["unk"] == 0:
                    r["dir"] = di
                    self.recs.append(r)
        self._mm: dict[tuple[int, str, int], np.memmap] = {}
        v = self.vocab.index
        self.BOS, self.EOS, self.PAD = v["<BOS>"], v["<EOS>"], v["<PAD>"]
        self.OBJECTS, self.CONT, self.CHAN = v["<OBJECTS>"], v["<CONT>"], v["<CHAN>"]
        self.LAYOUT, self.DECO, self.GAMEPLAY = v["<LAYOUT>"], v["<DECO>"], v["<GAMEPLAY>"]
        self.seg_lo = v[f"SEG:{SEG_RANGE[0]}"]
        self.seg_hi = self.seg_lo + SEG_RANGE[1] - SEG_RANGE[0]
        assert v[f"SEG:{SEG_RANGE[1] - 1}"] == self.seg_hi - 1, "SEG tokens must be contiguous"
        obj_ids = [i for t, i in v.items() if t.startswith("O:")] + [v["O?"]]
        self.is_obj = np.zeros(len(self.vocab), dtype=bool)
        self.is_obj[obj_ids] = True
        self.meta = {k: i for k, i in v.items() if k.startswith("M:")}

    def __len__(self) -> int:
        return len(self.recs)

    def stream(self, rec: dict, name: str) -> np.ndarray:
        key = (rec.get("dir", 0), name, rec["shard"])
        if key not in self._mm:
            path = self.dirs[key[0]] / f"{name}_{rec['shard']:03d}.bin"
            self._mm[key] = np.memmap(path, dtype=np.uint16, mode="r")
        off, n = rec[name]
        return self._mm[key][off:off + n]

    def meta_tokens(self, rec: dict) -> list[int]:
        m = self.meta
        return [m[f"M:stars={min(10, max(0, rec['stars']))}"], m[f"M:demon={min(6, rec['demon'])}"],
                m[f"M:epic={min(3, rec['epic'])}"], m[f"M:len={min(5, rec['length'])}"]]

    def header(self, rec: dict, max_len: int) -> np.ndarray:
        """Header tokens (<HEADER> .. last color channel), cut at a channel boundary to at most max_len."""
        h = self.stream(rec, "layout")[1:rec["header_len"] - 1]
        if len(h) <= max_len:
            return np.asarray(h)
        chans = np.flatnonzero(h[:max_len + 1] == self.CHAN)
        return np.asarray(h[:chans[-1] if len(chans) else max_len])

    def segments(self, objs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positions of SEG tokens in an object stream and the segment numbers they open."""
        pos = np.flatnonzero((objs >= self.seg_lo) & (objs < self.seg_hi))
        return pos, objs[pos].astype(np.int64) - self.seg_lo + SEG_RANGE[0]


class WindowSampler:
    def __init__(self, data: TokenData, ctx_len: int, p_layout: float = 0.4, p_fresh: float = 0.1,
                 header_max: int = 768, deco_prev: int = 512, seed: int = 0):
        self.d, self.ctx = data, ctx_len
        self.p_layout, self.p_fresh, self.header_max, self.deco_prev = p_layout, p_fresh, header_max, deco_prev
        self.rng = np.random.default_rng(seed)
        self.cum = {s: np.cumsum([r[s][1] for r in data.recs], dtype=np.float64) for s in ("layout", "deco")}

    def pick(self, stream: str) -> dict:
        c = self.cum[stream]
        return self.d.recs[int(np.searchsorted(c, self.rng.random() * c[-1], side="right"))]

    def sample(self, kind: str | None = None) -> tuple[np.ndarray, np.ndarray, str]:
        kind = kind or ("layout" if self.rng.random() < self.p_layout else "deco")
        if kind == "deco":
            out = self.deco_window(self.pick("deco"))
            if out is not None:
                return (*out, "deco")
        return (*self.layout_window(self.pick("layout")), "layout")

    def _finish(self, seq, loss_from: int) -> tuple[np.ndarray, np.ndarray]:
        """seq -> (inputs, targets); only tokens at sequence positions >= loss_from are predicted."""
        seq = np.asarray(seq, dtype=np.int64)[: self.ctx + 1]
        n = len(seq)
        full = np.full(self.ctx + 1, self.d.PAD, dtype=np.int64)
        full[:n] = seq
        tgt = full[1:].copy()
        tgt[: max(0, loss_from - 1)] = IGNORE
        tgt[n - 1:] = IGNORE
        return full[:-1], tgt

    def layout_window(self, rec: dict) -> tuple[np.ndarray, np.ndarray]:
        d = self.d
        objs = d.stream(rec, "layout")[rec["header_len"]:]
        prefix = [d.BOS, d.LAYOUT, *d.meta_tokens(rec)]
        header = d.header(rec, self.header_max)
        pos, _ = d.segments(objs)
        if len(pos) == 0 or self.rng.random() < self.p_fresh:
            return self._finish([*prefix, *header, d.OBJECTS, *objs[: self.ctx]], len(prefix))
        k = int(self.rng.integers(len(pos)))
        seq = [*prefix, *header, d.OBJECTS, d.CONT]
        loss_from = len(seq)
        seq += list(objs[pos[k]: pos[k] + self.ctx])
        return self._finish(seq, loss_from)

    def deco_window(self, rec: dict) -> tuple[np.ndarray, np.ndarray] | None:
        d = self.d
        lobjs = d.stream(rec, "layout")[rec["header_len"]:-1]
        dobjs = d.stream(rec, "deco")[1:]
        dpos, dseg = d.segments(dobjs)
        if len(dpos) == 0:
            return None
        lpos, lseg = d.segments(lobjs)
        k = int(self.rng.integers(len(dpos)))
        s = int(dseg[k])

        prev: list[int] = []
        if k > 0:
            tail = dobjs[dpos[k - 1] + 1: dpos[k]][-self.deco_prev:]
            starts = np.flatnonzero(d.is_obj[tail])
            prev = [int(dobjs[dpos[k - 1]]), *tail[starts[0]:]] if len(starts) else []
        head = [d.BOS, d.DECO, *d.meta_tokens(rec), *d.header(rec, self.header_max)]
        budget = self.ctx + 1 - len(head) - len(prev) - 3          # <GAMEPLAY> <OBJECTS> <CONT>

        def layout_slice(lo: int, hi: int) -> np.ndarray:
            a = int(np.searchsorted(lseg, lo, side="left"))
            b = int(np.searchsorted(lseg, hi, side="right"))
            return lobjs[lpos[a]: lpos[b] if b < len(lpos) else len(lobjs)] if a < len(lpos) else lobjs[:0]

        target = dobjs[dpos[k]:]
        t_len = min(len(target), int(budget * 0.75))
        last = k + len(np.flatnonzero((target[:t_len] >= d.seg_lo) & (target[:t_len] < d.seg_hi))) - 1
        gameplay = layout_slice(s - 1, int(dseg[min(last, len(dseg) - 1)]) + 2)
        if len(gameplay) + t_len > budget:
            gameplay = layout_slice(s - 1, s + 2)[: budget // 2]
            t_len = min(t_len, budget - len(gameplay))
        if t_len < 16:
            return None
        seq = [*head, d.GAMEPLAY, *gameplay, d.OBJECTS, *([d.CONT] if k > 0 else []), *prev]
        loss_from = len(seq)
        seq += list(target[:t_len])
        return self._finish(seq, loss_from)


class WindowDataset(torch.utils.data.IterableDataset):
    """Endless stream of windows; every DataLoader worker (and DDP rank) gets its own seed."""

    def __init__(self, root: str | Path, ctx_len: int, split: str = "train", seed: int = 0, rank: int = 0, **kw):
        self.root, self.ctx_len, self.split, self.seed, self.rank, self.kw = root, ctx_len, split, seed, rank, kw

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        sampler = WindowSampler(TokenData(self.root, self.split), self.ctx_len,
                                seed=self.seed * 1_000_003 + self.rank * 1009 + wid, **self.kw)
        while True:
            x, y, _ = sampler.sample()
            yield torch.from_numpy(x), torch.from_numpy(y)
