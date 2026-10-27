"""Decoder-only transformer over level tokens (gdgen/tokens.py).

RoPE, RMSNorm, SwiGLU, grouped-query attention through torch SDPA (FlashAttention on H100), tied embeddings.
The loss is computed in position chunks under activation checkpointing, so a 65k-token vocabulary does not need
batch x context x vocab logits in memory. `generate` keeps a KV cache; `BatchDecoder` samples many requests at once
(continuous batching over a preallocated per-row KV cache), which is what makes generation fast: one token of one
sequence costs as much GPU time as one token of eight.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

PRESETS = {
    "test": dict(n_layer=2, n_head=4, n_kv_head=2, d_model=64, ctx_len=512),
    "s": dict(n_layer=12, n_head=12, n_kv_head=4, d_model=768),        # ~85M + embeddings
    "m": dict(n_layer=24, n_head=16, n_kv_head=4, d_model=1024),       # ~300M + embeddings
    "l": dict(n_layer=32, n_head=20, n_kv_head=4, d_model=1280),       # ~640M + embeddings
}


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int = 24
    n_head: int = 16
    n_kv_head: int = 4
    d_model: int = 1024
    ctx_len: int = 8192
    rope_theta: float = 500_000.0
    mlp_hidden: int = 0                 # 0 -> 8/3 d_model rounded up to 128
    loss_chunk: int = 2048

    def __post_init__(self):
        if not self.mlp_hidden:
            self.mlp_hidden = 128 * math.ceil(self.d_model * 8 / 3 / 128)

    @classmethod
    def preset(cls, name: str, vocab_size: int, **overrides) -> "GPTConfig":
        return cls(vocab_size=vocab_size, **{**PRESETS[name], **overrides})

    def to_dict(self) -> dict:
        return asdict(self)


def rope_cache(head_dim: int, length: int, theta: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    ang = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); cos/sin: (T, D/2), or (B, D/2) for one token per row at its own position (T = 1)
    x1, x2 = x.float().chunk(2, dim=-1)
    cos, sin = (cos[:, None, None], sin[:, None, None]) if x.shape[2] == 1 and cos.shape[0] == x.shape[0] \
        else (cos[None, None], sin[None, None])
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head, self.n_kv, self.hd = cfg.n_head, cfg.n_kv_head, cfg.d_model // cfg.n_head
        self.q = nn.Linear(cfg.d_model, cfg.n_head * self.hd, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.n_kv_head * self.hd, bias=False)
        self.o = nn.Linear(cfg.n_head * self.hd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache=None):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_head, self.hd).transpose(1, 2)
        k, v = self.kv(x).view(b, t, 2, self.n_kv, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        if q.shape[2] == k.shape[2]:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        elif q.shape[2] == 1:
            y = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        else:
            # prefill on top of an existing cache: queries see the whole cache plus their own causal prefix
            past = k.shape[2] - q.shape[2]
            mask = torch.ones(q.shape[2], k.shape[2], dtype=torch.bool, device=x.device).tril(past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        return self.o(y.transpose(1, 2).reshape(b, t, -1))

    def step(self, x, cos, sin, kc, vc, pos, mask):
        """One token per row: x (B, 1, C), cos/sin (B, D/2) at the rows' positions pos (B,), written into the row
        caches kc/vc (B, Hkv, ctx, D); mask (B, 1, 1, T) marks each row's valid keys."""
        b = x.shape[0]
        q = self.q(x).view(b, 1, self.n_head, self.hd).transpose(1, 2)
        k, v = self.kv(x).view(b, 1, 2, self.n_kv, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rows = torch.arange(b, device=x.device)
        kc[rows, :, pos] = k[:, :, 0].to(kc.dtype)
        vc[rows, :, pos] = v[:, :, 0].to(vc.dtype)
        t = mask.shape[-1]
        y = F.scaled_dot_product_attention(q, kc[:, :, :t].to(q.dtype), vc[:, :, :t].to(q.dtype), attn_mask=mask,
                                           enable_gqa=True)
        return self.o(y.transpose(1, 2).reshape(b, 1, -1))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, 2 * cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        a, g = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * a)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
        self.attn, self.mlp = Attention(cfg), MLP(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.n1(x), cos, sin, cache)
        return x + self.mlp(self.n2(x))

    def step(self, x, cos, sin, kc, vc, pos, mask):
        x = x + self.attn.step(self.n1(x), cos, sin, kc, vc, pos, mask)
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.RMSNorm(cfg.d_model)
        cos, sin = rope_cache(cfg.d_model // cfg.n_head, cfg.ctx_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.attn.o.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))
            nn.init.normal_(b.mlp.down.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.emb.weight.numel() if non_embedding else n

    def hidden(self, idx: torch.Tensor, start: int = 0, caches=None, grad_ckpt: bool = False) -> torch.Tensor:
        t = idx.shape[1]
        cos, sin = self.cos[start:start + t], self.sin[start:start + t]
        x = self.emb(idx)
        for i, blk in enumerate(self.blocks):
            if grad_ckpt and self.training:
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin, None if caches is None else caches[i])
        return self.norm(x)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.emb.weight.T

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, grad_ckpt: bool = False):
        """(summed loss over targets != -100, number of counted targets), or logits when no targets are given."""
        h = self.hidden(idx, grad_ckpt=grad_ckpt)
        if targets is None:
            return self.logits(h)
        total = h.new_zeros((), dtype=torch.float32)
        for s in range(0, h.shape[1], self.cfg.loss_chunk):
            hc, tc = h[:, s:s + self.cfg.loss_chunk], targets[:, s:s + self.cfg.loss_chunk]
            total = total + checkpoint(self._chunk_loss, hc, tc, use_reentrant=False)
        return total, (targets != -100).sum()

    def _chunk_loss(self, h, t):
        return F.cross_entropy(self.logits(h).float().flatten(0, 1), t.flatten(), ignore_index=-100, reduction="sum")

    @torch.no_grad()
    def step(self, tokens: torch.Tensor, pos: torch.Tensor, cache: "RowCache") -> torch.Tensor:
        """Logits (B, V) after feeding tokens[b] at position pos[b] of row b of the cache."""
        x = self.emb(tokens[:, None])
        cos, sin = self.cos[pos], self.sin[pos]
        t = int(pos.max()) + 1
        mask = (torch.arange(t, device=tokens.device)[None, :] <= pos[:, None])[:, None, None, :]
        for blk, kc, vc in zip(self.blocks, cache.k, cache.v):
            x = blk.step(x, cos, sin, kc, vc, pos, mask)
        return self.logits(self.norm(x))[:, 0].float()

    @torch.no_grad()
    def prefill_row(self, prompt: list[int], cache: "RowCache", row: int) -> torch.Tensor:
        """Runs a prompt into one row of the cache; returns the logits (V,) of its last position."""
        caches = [{} for _ in self.blocks]
        h = self.hidden(torch.tensor([prompt], device=self.emb.weight.device), 0, caches)
        n = len(prompt)
        for c, kc, vc in zip(caches, cache.k, cache.v):
            kc[row, :, :n] = c["k"][0].to(kc.dtype)
            vc[row, :, :n] = c["v"][0].to(vc.dtype)
        return self.logits(h[:, -1]).float()[0]

    @torch.no_grad()
    def generate(self, prompt: list[int], max_new: int, temperature: float = 1.0, top_p: float = 0.95,
                 allowed=None, stop_ids: set[int] | None = None, generator: torch.Generator | None = None,
                 inject=None, until=None) -> list[int]:
        """Samples up to max_new tokens after prompt. `allowed(tokens) -> bool tensor | None` can mask the vocabulary
        (grammar constraints); `inject(tokens) -> list[int]` returns tokens forced after the one just sampled (music
        after a SEG); sampling stops at a token in stop_ids, when `until(tokens)` is true right after a sampled token
        (the token stays in the output) or when the context is full."""
        dev = self.emb.weight.device
        caches = [{} for _ in self.blocks]
        toks = list(prompt)
        h = self.hidden(torch.tensor([toks], device=dev), 0, caches)
        out = []
        for _ in range(max_new):
            logits = self.logits(h[:, -1]).float()[0]
            if allowed is not None:
                m = allowed(toks)
                if m is not None:
                    logits = logits.masked_fill(~m.to(dev), float("-inf"))
            nxt = sample_token(logits, temperature, top_p, generator)
            toks.append(nxt)
            out.append(nxt)
            forced = list(inject(toks)) if inject is not None else []      # the injector sees every sampled token
            if (stop_ids and nxt in stop_ids) or (until is not None and until(toks)) or len(toks) >= self.cfg.ctx_len:
                break
            forced = forced[: self.cfg.ctx_len - len(toks)]
            toks += forced
            out += forced
            if len(toks) >= self.cfg.ctx_len:
                break
            h = self.hidden(torch.tensor([[nxt, *forced]], device=dev), len(toks) - 1 - len(forced), caches)
        return out


def sample_token(logits: torch.Tensor, temperature: float, top_p: float, generator=None) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sp, si = probs.sort(descending=True)
        keep = sp.cumsum(0) - sp < top_p
        probs = torch.zeros_like(probs).scatter(0, si[keep], sp[keep])
    return int(torch.multinomial(probs, 1, generator=generator))


class RowCache:
    """Preallocated keys/values for `rows` sequences of up to ctx_len tokens (one tensor pair per layer)."""

    def __init__(self, model: GPT, rows: int, dtype: torch.dtype | None = None):
        cfg, w = model.cfg, model.emb.weight
        shape = (rows, cfg.n_kv_head, cfg.ctx_len, cfg.d_model // cfg.n_head)
        self.k = [torch.zeros(shape, device=w.device, dtype=dtype or w.dtype) for _ in model.blocks]
        self.v = [torch.zeros(shape, device=w.device, dtype=dtype or w.dtype) for _ in model.blocks]


@dataclass
class SampleRequest:
    """What GPT.generate takes, as a value: sampling requests that BatchDecoder runs side by side."""
    prompt: list[int]
    max_new: int
    temperature: float = 1.0
    top_p: float = 0.95
    allowed: Callable | None = None
    stop_ids: set[int] | None = None
    generator: torch.Generator | None = None
    inject: Callable | None = None
    until: Callable | None = None


@dataclass
class _Row:
    task: int
    req: SampleRequest
    toks: list[int]
    out: list[int] = field(default_factory=list)
    forced: list[int] = field(default_factory=list)
    sampled: int = 0
    logits: torch.Tensor | None = None


def run_sync(task, model: GPT):
    """Drives one task (a generator yielding SampleRequests, see BatchDecoder) with GPT.generate; returns its value."""
    try:
        req = task.send(None)
        while True:
            req = task.send(model.generate(req.prompt, req.max_new, req.temperature, req.top_p, allowed=req.allowed,
                                           stop_ids=req.stop_ids, generator=req.generator, inject=req.inject,
                                           until=req.until))
    except StopIteration as e:
        return e.value


class BatchDecoder:
    """Continuous batching: tasks are generators that yield SampleRequests and get each result (the new tokens, as
    GPT.generate returns them) sent back; up to `rows` requests decode together, one token per row per model step.
    A request's tokens follow GPT.generate's rules: stop at a stop id, at `until`, at max_new sampled tokens or a full
    context; tokens from `inject` are fed one per step instead of sampling."""

    def __init__(self, model: GPT, rows: int, dtype: torch.dtype | None = None):
        self.model, self.rows = model, rows
        self.cache = RowCache(model, rows, dtype)

    def run(self, tasks: list) -> list:
        results: list = [None] * len(tasks)
        waiting: deque = deque()
        for i, task in enumerate(tasks):
            self._resume(i, tasks, None, waiting, results)
        rows: list[_Row | None] = [None] * self.rows
        dev = self.model.emb.weight.device
        ctx = self.model.cfg.ctx_len
        while waiting or any(r is not None for r in rows):
            for r in range(self.rows):
                if rows[r] is None and waiting:
                    i, req = waiting.popleft()
                    rows[r] = _Row(i, req, list(req.prompt), logits=self.model.prefill_row(req.prompt, self.cache, r))
            feed = torch.zeros(self.rows, dtype=torch.long)
            pos = torch.zeros(self.rows, dtype=torch.long)
            for r, st in enumerate(rows):
                if st is None:
                    continue
                req = st.req
                if st.forced:
                    nxt, sampled = st.forced.pop(0), False
                else:
                    logits = st.logits
                    if req.allowed is not None:
                        m = req.allowed(st.toks)
                        if m is not None:
                            logits = logits.masked_fill(~m.to(logits.device), float("-inf"))
                    nxt, sampled = sample_token(logits, req.temperature, req.top_p, req.generator), True
                    st.sampled += 1
                st.toks.append(nxt)
                st.out.append(nxt)
                done = len(st.toks) >= ctx
                if sampled:
                    forced = list(req.inject(st.toks)) if req.inject is not None else []
                    if (req.stop_ids and nxt in req.stop_ids) or (req.until is not None and req.until(st.toks)):
                        done = True
                    elif not done:
                        st.forced = forced[: ctx - len(st.toks)]
                        if st.sampled >= req.max_new:      # as GPT.generate: the last token's forced ones still go in
                            st.toks += st.forced
                            st.out += st.forced
                            st.forced, done = [], True
                if done:
                    rows[r] = None
                    self._resume(st.task, tasks, st.out, waiting, results)
                    continue
                feed[r], pos[r] = nxt, len(st.toks) - 1
            active = [r for r, st in enumerate(rows) if st is not None and st.logits is not None]
            if not active:
                continue
            logits = self.model.step(feed.to(dev), pos.to(dev), self.cache)
            for r in active:
                rows[r].logits = logits[r]
        return results

    @staticmethod
    def _resume(i: int, tasks: list, value, waiting: deque, results: list) -> None:
        try:
            waiting.append((i, tasks[i].send(value)))
        except StopIteration as e:
            results[i] = e.value
