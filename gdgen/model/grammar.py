"""Next-token masks that keep sampled streams well-formed (used as `allowed` in GPT.generate).

Header (when the model writes it): <HEADER> (H:k value)* <COLORS> (<CHAN> (CP:p value)*)* <OBJECTS>, with the H keys
and the CP properties of one channel strictly increasing.
Objects: O X Y YF, then attributes in the encoder's order - flags (F:k increasing, or one FS set), explicit zeros
(F0:k increasing), groups (G:*), other keys (A:k=v / K:k increasing) - each key at most once; segments only move
forward; a <LAYOUT> stream holds gameplay objects only.
Values follow the key's kind exactly: one value unit (V:x | EMPTY | D:c.. DEND | TRAW B:.. LEND) for numbers,
units until LEND for lists, B:.. LEND for text, HH HS HV for HSV. Padding ids (the model vocabulary rounded up) are
never allowed.
"""
from __future__ import annotations

import numpy as np
import torch

from gdgen.objects import GAMEPLAY_CATEGORIES, category_of
from gdgen.tokens import COLOR_HSV_PROP, SEG_RANGE, Y_RANGE, Vocab

(OTHER, SEG, OBJ, X, Y, YF, ATTR, VAL, END, START, HEAD, HKEY, COLORS, CHAN, CPROP) = range(15)
GROUP_RANK = 30_000


def attr_rank(t: str) -> int:
    """Position of an attribute token in the encoder's order (see Tokenizer.encode_object)."""
    if t.startswith("FS:"):
        return 19_999
    if t.startswith("F:"):
        return 10_000 + int(t[2:])
    if t.startswith("F0:"):
        return 20_000 + int(t[3:])
    if t.startswith("G:"):
        return GROUP_RANK
    key = t[2:].split("=", 1)[0]
    return 40_000 + (128 if key == "s" else int(key))


def value_mode(kind: str) -> str:
    """How the value of a K:<key> of this kind is spelled (see Tokenizer.attribute)."""
    return {"list": "list", "astr": "list", "groups": "list", "text": "text", "hsv": "hsv"}.get(kind, "single")


class Grammar:
    def __init__(self, vocab: Vocab, size: int | None = None, consecutive: bool = False, max_skip: int | None = None,
                 seg_cap: int | None = None, max_same_x: int | None = None):
        """size: the model's (padded) vocabulary size; ids beyond the real vocabulary are never allowed.
        consecutive: segments may not be skipped (with music, every segment carries its part of the song).
        max_skip: at most this many empty segments in a row (featured levels have almost none).
        seg_cap: after this many tokens in one segment only the next SEG or <EOS> may follow an object (stops a
        sample from piling thousands of objects into one segment).
        max_same_x: after this many objects in a row at one x the next object must lie further right (a sampled
        copy-the-last-object loop otherwise stacks a tower of identical blocks)."""
        self.consecutive, self.max_skip, self.seg_cap, self.max_same_x = consecutive, max_skip, seg_cap, max_same_x
        n = size or len(vocab)
        cat = np.full(n, OTHER, dtype=np.int8)
        rank = np.full(n, -1, dtype=np.int64)
        self.k_mode: dict[int, str] = {}
        layout_objs = []
        for t, i in vocab.index.items():
            if t.startswith("SEG:"):
                cat[i] = SEG
            elif t.startswith("O:") or t == "O?":
                cat[i] = OBJ
                if t == "O?" or category_of(int(t[2:])) in GAMEPLAY_CATEGORIES - {"start_pos"}:
                    layout_objs.append(i)
            elif t.startswith("X:"):
                cat[i] = X
            elif t.startswith("YF:"):
                cat[i] = YF
            elif t.startswith("Y:"):
                cat[i] = Y
            elif t.startswith(("FS:", "F:", "F0:", "G:", "A:", "K:")):
                cat[i], rank[i] = ATTR, attr_rank(t)
                if t.startswith("K:"):
                    self.k_mode[i] = value_mode(vocab.key_kinds.get(t[2:], "num"))
            elif t.startswith(("V:", "D:", "B:", "HH:", "HS:", "HV:")) or t in ("DEND", "TRAW", "LEND", "EMPTY"):
                cat[i] = VAL
            elif t == "<EOS>":
                cat[i] = END
            elif t in ("<OBJECTS>", "<CONT>"):
                cat[i] = START
            elif t == "<HEADER>":
                cat[i] = HEAD
            elif t.startswith("H:k"):
                cat[i], rank[i] = HKEY, (0 if t[3] == "A" else 1000) + int(t[4:])
            elif t == "<COLORS>":
                cat[i] = COLORS
            elif t == "<CHAN>":
                cat[i] = CHAN
            elif t.startswith("CP:"):
                cat[i], rank[i] = CPROP, int(t[3:])
        self.cat, self.rank = cat, torch.from_numpy(rank)
        self.tokens = vocab.tokens
        v = vocab.index
        self.seg_lo = v[f"SEG:{SEG_RANGE[0]}"]
        # objects are sorted by (x, y cell, y fine) inside a segment (Tokenizer.encode_objects): positions never go back
        self.x_lo, self.y_lo, self.yf_lo = v["X:0"], v[f"Y:{Y_RANGE[0]}"], v["YF:0"]
        self.oq = v["O?"]
        self.layout_tok, self.deco_tok = v["<LAYOUT>"], v["<DECO>"]
        self.cp_hsv = v[f"CP:{COLOR_HSV_PROP}"]

        def m(*cats):
            return torch.from_numpy(np.isin(cat, cats))

        def prefixed(*prefixes, exact=()):
            out = torch.zeros(n, dtype=torch.bool)
            out[[i for t, i in v.items() if t.startswith(prefixes) or t in exact]] = True
            return out

        self.is_attr, self.is_g = m(ATTR), prefixed("G:")
        self.is_seg_tail = m(SEG)                    # used to drop SEG tokens beyond the next one
        self.unit_start = prefixed("V:", "D:", exact=("EMPTY", "TRAW"))
        self.digits = prefixed("D:", exact=("DEND",))
        self.bytes = prefixed("B:", exact=("LEND",))
        self.lend = prefixed(exact=("LEND",))
        self.hh, self.hs, self.hv = prefixed("HH:"), prefixed("HS:"), prefixed("HV:")
        self.after_obj, self.after_x, self.after_y = m(X), m(Y), m(YF)
        self.attrs_or_next = m(ATTR, OBJ, SEG, END)
        self.between = m(OBJ, SEG, END)
        self.seg_or_end = m(SEG, END)
        self.layout_only = torch.ones(n, dtype=torch.bool)
        self.layout_only[torch.from_numpy(cat == OBJ)] = False
        self.layout_only[layout_objs] = True
        self.is_hkey, self.is_cprop, self.is_colors = m(HKEY), m(CPROP), m(COLORS)
        self.pre = m(HEAD) | prefixed("M:", exact=("<FULL>", "<LAYOUT>", "<DECO>"))   # <BOS> [stream] M:* <HEADER>
        self.only_head = m(HEAD)
        self.chan_or_objects = m(CHAN)
        self.chan_or_objects[v["<OBJECTS>"]] = True
        self.real = torch.zeros(n, dtype=torch.bool)
        self.real[: len(vocab)] = True
        self.reset()

    def reset(self) -> None:
        self.region, self.stream, self.seg, self.last, self.seen = "pre", None, SEG_RANGE[0] - 1, -1, 0
        self.since_seg = 0                          # tokens since the last SEG
        self.prev_pos = None                        # (x, y cell, y fine) of the segment's last object
        self.cur_x = self.cur_y = None
        self.same_x = 0                             # objects in a row at prev_pos's x
        self.vmode, self.vpos = None, None          # value being written: mode single|list|text|hsv, position
        self.expect_x = False                       # inside "O[?] value" before X

    # ---- state

    def _start_value(self, mode: str) -> None:
        self.vmode, self.vpos = mode, ("bytes" if mode == "text" else "start")

    def _unit_done(self) -> None:
        if self.vmode == "list":
            self.vpos = "start"
        else:
            self.vmode = self.vpos = None

    def _unit_start(self, tok: str) -> None:
        if tok.startswith("D:"):
            self.vpos = "digits"
        elif tok == "TRAW":
            self.vpos = "bytes"
        else:                                                         # V:x or EMPTY
            self._unit_done()

    def _value_token(self, tok: str) -> None:
        if self.vpos == "digits":
            if tok == "DEND":
                self._unit_done()
        elif self.vpos == "bytes":
            if tok == "LEND":
                if self.vmode == "text":
                    self.vmode = self.vpos = None
                else:
                    self._unit_done()
            elif not tok.startswith("B:"):                            # text that fell back to a plain value
                self.vmode = "single"
                self._unit_start(tok)
        elif self.vpos == "hs":
            self.vpos = "hv"
        elif self.vpos == "hv":
            self.vmode = self.vpos = None
        elif tok == "LEND":                                           # end of a list
            self.vmode = self.vpos = None
        elif tok.startswith("HH:"):
            self.vpos = "hs"
        else:
            if self.vmode == "hsv":
                self.vmode = "list"                                   # HSV that fell back to 'a'-separated values
            self._unit_start(tok)

    def _advance(self, tokens: list[int]) -> None:
        for t in tokens[self.seen:]:
            c = self.cat[t]
            self.since_seg += 1
            if self.vmode is not None and c == VAL:
                self._value_token(self.tokens[t])
                continue
            self.vmode = self.vpos = None
            if t == self.layout_tok:
                self.stream = "layout"
            elif t == self.deco_tok:
                self.stream = "deco"
            elif c == HEAD:
                self.region, self.last = "header", -1
            elif c == COLORS:
                self.region, self.last = "colors", -1
            elif c == CHAN:
                self.last = -1
            elif c == START:
                self.region, self.last = "objects", -1
            elif c == SEG:
                self.seg = int(t) - self.seg_lo + SEG_RANGE[0]
                self.since_seg, self.prev_pos, self.same_x = 0, None, 0
            elif c == OBJ:
                self.last, self.expect_x = -1, True
                if t == self.oq:
                    self._start_value("single")
            elif c == X:
                self.expect_x, self.cur_x = False, int(t) - self.x_lo
            elif c == Y:
                self.cur_y = int(t) - self.y_lo
            elif c == YF:
                same = self.prev_pos is not None and self.prev_pos[0] == self.cur_x
                self.same_x = self.same_x + 1 if same else 1
                self.prev_pos = (self.cur_x, self.cur_y, int(t) - self.yf_lo)
            elif c == HKEY:
                self.last = int(self.rank[t])
                self._start_value("single")
            elif c == CPROP:
                self.last = int(self.rank[t])
                self._start_value("hsv" if t == self.cp_hsv else "single")
            elif c == ATTR:
                self.last = int(self.rank[t])
                if t in self.k_mode:
                    self._start_value(self.k_mode[t])
        self.seen = len(tokens)

    # ---- masks

    def _value_mask(self) -> torch.Tensor:
        if self.vpos == "digits":
            return self.digits
        if self.vpos == "bytes":
            return self.bytes | self.unit_start if self.vmode == "text" else self.bytes
        if self.vpos == "hs":
            return self.hs
        if self.vpos == "hv":
            return self.hv
        if self.vmode == "hsv":
            return self.hh | self.unit_start
        if self.vmode == "list":
            return self.unit_start | self.lend
        return self.unit_start

    def allowed(self, tokens: list[int]) -> torch.Tensor:
        if len(tokens) < self.seen:
            self.reset()
        self._advance(tokens)
        if self.vmode is not None:
            return self._value_mask()
        last = tokens[-1]
        c = self.cat[last]
        if self.region == "pre":
            if c != OTHER:
                return self.real
            return self.only_head if self.tokens[last].startswith("M:") else self.pre   # metadata is the prompt's
        if self.region == "header":
            if c == HEAD:
                return self.is_hkey
            return (self.is_hkey & (self.rank > self.last)) | self.is_colors
        if self.region == "colors":
            if c == CHAN:
                return self.is_cprop
            if c == COLORS:
                return self.chan_or_objects
            return (self.is_cprop & (self.rank > self.last)) | self.chan_or_objects
        prev = self.prev_pos
        if self.expect_x:
            if prev is None:
                return self.after_obj
            mask = self.after_obj.clone()
            full = self.max_same_x is not None and self.same_x >= self.max_same_x
            mask[self.x_lo: self.x_lo + prev[0] + (1 if full else 0)] = False
            return mask
        if c == X:
            if prev is None or self.cur_x != prev[0]:
                return self.after_x
            mask = self.after_x.clone()
            mask[self.y_lo: self.y_lo + prev[1]] = False
            return mask
        if c == Y:
            if prev is None or (self.cur_x, self.cur_y) != prev[:2]:
                return self.after_y
            mask = self.after_y.clone()
            mask[self.yf_lo: self.yf_lo + prev[2]] = False
            return mask
        if c in (YF, ATTR, VAL):
            mask = self.attrs_or_next.clone()
            stale = self.is_attr & (self.rank <= self.last)
            if self.last <= GROUP_RANK:
                stale &= ~self.is_g                                   # several groups in a row
            mask &= ~stale
        else:
            mask = self.between.clone()
        mask[self.seg_lo: self.seg_lo + (self.seg - SEG_RANGE[0]) + 1] = False     # segments only move forward
        skip = 0 if self.consecutive else self.max_skip
        if skip is not None and self.seg >= SEG_RANGE[0]:
            lo = self.seg_lo + (self.seg - SEG_RANGE[0]) + 2 + skip
            mask[lo:] &= ~self.is_seg_tail[lo:]
        if self.seg_cap is not None and self.since_seg > self.seg_cap:
            mask &= self.seg_or_end
        if self.stream == "layout":
            mask &= self.layout_only
        return mask
