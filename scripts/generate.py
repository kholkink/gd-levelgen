"""Generates a level with a trained checkpoint: the layout (gameplay) first, then decoration window by window.

    python scripts/generate.py --ckpt runs/m1/ckpt.pt --out gen/try1 [--stars 8 --demon 0 --epic 1 --length 3]
        [--temperature 0.9 --top-p 0.95 --seed 0] [--max-layout-tokens 60000] [--deco-window 2] [--no-deco]
    python scripts/generate.py --random-init --out gen/smoke      # pipeline check with an untrained tiny model

Layout: <BOS> <LAYOUT> M:* is continued until <EOS>; when the context fills up the next prompt is the header plus
the last half-context of objects (from a segment boundary) after <CONT>, exactly like the training windows.
Decoration: for segments s, s+W, ... the prompt is header + <GAMEPLAY> layout of segments s-1 .. s+W+1 +
<OBJECTS> [<CONT> previous deco tail] SEG:s, sampled until a SEG >= s+W or <EOS>.
Selection (--select-segments K): the layout is sampled K segments at a time; after each chunk the simulator searches a
path through the layout so far up to the chunk's end, an impassable chunk is resampled (--select-tries), and when no
try passes the previous chunk is resampled too (at most --select-backtracks times per level).
With --song-features the song's music tokens are forced after every SEG (gdgen/music.py MusicInjector).
--samples N --batch B generates N levels (seeds seed..seed+N-1, into <out>/s<seed>/) in one process, B sampling
requests decoding together (gdgen/model/gpt.py BatchDecoder): one model step serves all of them, so a batch of 8 is
several times faster than 8 processes. Each level is a coroutine (Generator.sample yields its requests).

Writes to --out: level.txt (raw level text), level_string.txt (base64 gzip, what the game stores),
tokens_layout.txt / tokens_deco.txt, report.json (counts, decode errors, simulator result on the layout).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gdgen.levelstring import encode_level_string, serialize_level  # noqa: E402
from gdgen.model.grammar import OBJ, Grammar  # noqa: E402
from gdgen.music import MusicInjector, level_music, load_features  # noqa: E402
from gdgen.model.gpt import GPT, BatchDecoder, GPTConfig, SampleRequest, run_sync  # noqa: E402
from gdgen.evaluate import simulate  # noqa: E402
from gdgen.tokens import SEG_RANGE, VOCAB_FILE, Tokenizer, Vocab  # noqa: E402


class Generator:
    def __init__(self, model: GPT, vocab: Vocab, args, seed: int | None = None, tag: str = ""):
        self.model, self.vocab, self.args, self.tag = model, vocab, args, tag
        self.tok = Tokenizer(vocab)
        self.v = vocab.index
        self.features = load_features(args.song_features) if args.song_features else None
        self.grammar = Grammar(vocab, model.cfg.vocab_size, consecutive=self.features is not None,
                               max_skip=args.max_seg_skip, max_same_x=args.max_same_x)
        self.deco_music: dict[int, list[str]] = {}
        self.selection: dict | None = None
        self.gen = torch.Generator(device=model.emb.weight.device).manual_seed(args.seed if seed is None else seed)
        self.ctx = model.cfg.ctx_len
        self.seg_lo = self.v[f"SEG:{SEG_RANGE[0]}"]
        self.meta = [self.v[f"M:stars={args.stars}"], self.v[f"M:demon={args.demon}"],
                     self.v[f"M:epic={args.epic}"], self.v[f"M:len={args.length}"]]

    def seg_of(self, t: int) -> int | None:
        s = t - self.seg_lo
        return s + SEG_RANGE[0] if 0 <= s < SEG_RANGE[1] - SEG_RANGE[0] else None

    def seg_token(self, s: int) -> int:
        return self.seg_lo + s - SEG_RANGE[0]

    def sample(self, prompt: list[int], max_new: int, stop: set[int], inject=None, until=None):
        """Coroutine step: yields the sampling request, gets the new tokens back (`x = yield from self.sample(...)`)."""
        self.grammar.reset()
        max_new = min(max_new, self.ctx - len(prompt) - 1)
        if max_new <= 0:
            return []
        return (yield SampleRequest(prompt, max_new, self.args.temperature, self.args.top_p, self.grammar.allowed,
                                    stop, self.gen, inject, until))

    def music_ids(self, tokens: list[str]) -> list[int]:
        return [self.v[t] for t in tokens]

    def header_trunc(self, header: list[int]) -> list[int]:
        if len(header) <= self.args.header_max:
            return header
        chans = [i for i, t in enumerate(header[: self.args.header_max + 1]) if t == self.v["<CHAN>"]]
        return header[: chans[-1] if chans else self.args.header_max]

    def tail_from_segment(self, objs: list[int], keep: int) -> list[int]:
        """Context for the next window: the last <= keep tokens of objs from a segment boundary, or, when there is
        none, from an object start led by the SEG it belongs to (so the segment never goes backwards)."""
        start = max(0, len(objs) - keep)
        for i in range(start, len(objs)):
            if self.seg_of(objs[i]) is not None:
                return objs[i:]
        i = next((j for j in range(start, len(objs)) if self.grammar.cat[objs[j]] == OBJ), None)
        if i is None:
            return []
        seg = next((objs[j] for j in range(i - 1, -1, -1) if self.seg_of(objs[j]) is not None), None)
        return ([seg] if seg is not None else []) + objs[i:]

    def last_seg(self, toks: list[int]) -> int | None:
        for t in reversed(toks):
            s = self.seg_of(t)
            if s is not None:
                return s
        return None

    def layout(self) -> tuple[list[int], list[int], bool]:
        eos, start = self.v["<EOS>"], self.v["<OBJECTS>"]
        self.grammar.seg_cap = self.args.layout_seg_cap
        prompt = [self.v["<BOS>"], self.v["<LAYOUT>"], *self.meta]
        # training headers are at most header_max tokens (cut at a color channel): close an overlong one the same way
        header = yield from self.sample(prompt, self.args.header_max + 1, {start})
        header = header[:-1] if header and header[-1] == start else self.header_trunc(header)
        inject = None
        if self.features is not None:
            head, _ = self.tok.decode_level(self.vocab.detokenize([self.v["<BOS>"], *header, start, eos]), strict=False)
            inject = MusicInjector(self.tok, self.features, head, self.args.song_offset)
        if self.args.select_segments:
            objs, finished = yield from self.layout_selected(header, inject)
            return header, objs, finished
        objs = self.with_music((yield from self.sample([*prompt, *header, start], self.ctx, {eos}, inject)), inject)
        while (not objs or objs[-1] != eos) and len(objs) < self.args.max_layout_tokens:
            tail = self.tail_from_segment(objs, self.ctx // 2)
            prompt = [self.v["<BOS>"], self.v["<LAYOUT>"], *self.meta, *self.header_trunc(header),
                      self.v["<OBJECTS>"], self.v["<CONT>"], *tail]
            new = self.with_music((yield from self.sample(prompt, self.ctx, {eos}, inject)), inject)
            if not new:
                break
            objs += new
            print(f"{self.tag}  layout: {len(objs)} tokens, segment {self.last_seg(objs)}", flush=True)
        finished = bool(objs) and objs[-1] == eos
        return header, objs[:-1] if finished else objs, finished

    def layout_prompt(self, header: list[int], objs: list[int]) -> list[int]:
        """Header + objects so far, or + <CONT> and the last half-context of them once they fill half the context."""
        base = [self.v["<BOS>"], self.v["<LAYOUT>"], *self.meta, *self.header_trunc(header), self.v["<OBJECTS>"]]
        if len(base) + len(objs) <= self.ctx // 2:
            return base + objs
        return base + [self.v["<CONT>"], *self.tail_from_segment(objs, self.ctx // 2)]

    def check(self, header: list[int], objs: list[int], goal_x: float | None) -> dict:
        names = self.vocab.detokenize([self.v["<BOS>"], *header, self.v["<OBJECTS>"], *objs, self.v["<EOS>"]])
        head, gameplay = self.tok.decode_level(names, strict=False)
        if not gameplay:
            return {"passed": True, "max_x": 0.0, "seconds": 0.0}
        try:
            return simulate(head, gameplay, self.args.max_frontier, self.args.select_sim_seconds, goal_x)
        except Exception as e:  # noqa: BLE001  (a malformed chunk counts as impassable)
            return {"passed": False, "max_x": 0.0, "seconds": 0.0, "error": f"{type(e).__name__}: {e}"}

    def layout_selected(self, header: list[int], inject: MusicInjector | None) -> tuple[list[int], bool]:
        """Layout in chunks of --select-segments segments, each resampled until the simulator passes it."""
        a, eos = self.args, self.v["<EOS>"]
        chunks: list[tuple[list[int], tuple | None]] = []        # accepted tokens + injector state after them
        start_state = inject.state() if inject else None
        st = self.selection = {"chunks": 0, "attempts": 0, "rejected": 0, "backtracks": 0, "unresolved_at": None,
                               "sim_seconds": 0.0}
        backtracks, checking, finished = a.select_backtracks, True, False
        while not finished:
            objs = [t for c, _ in chunks for t in c]
            if len(objs) >= a.max_layout_tokens:
                break
            base_state = chunks[-1][1] if chunks else start_state
            best = None
            for _ in range(a.select_tries if checking else 1):
                if inject:
                    inject.restore(base_state)
                stop = ChunkStop(self, a.select_segments)
                new = yield from self.sample(self.layout_prompt(header, objs), self.ctx, {eos}, inject, stop)
                done = bool(new) and new[-1] == eos
                cand = new[:-1] if done or stop.fired else self.with_music(new, inject)
                st["attempts"] += 1
                res = self.check(header, objs + cand, None if done else stop.goal_x) if checking else {"passed": True}
                st["sim_seconds"] += res.get("seconds", 0.0)
                if best is None or res["passed"] or res.get("max_x", 0) > best[3].get("max_x", 0):
                    best = (cand, inject.state() if inject else None, done, res)
                if res["passed"]:
                    break
                st["rejected"] += 1
            cand, state, done, res = best
            if not res["passed"]:
                if backtracks > 0 and chunks:              # the cause may sit in the previous chunk: redo it as well
                    chunks.pop()
                    backtracks -= 1
                    st["backtracks"] += 1
                    continue
                st["unresolved_at"] = res.get("max_x")     # keep the furthest try, stop checking
                checking = False
            chunks.append((cand, state))
            st["chunks"] = len(chunks)
            finished = done
            print(f"{self.tag}  layout: chunk {len(chunks)}, segment {self.last_seg(objs + cand)}, "
                  f"{st['attempts']} tries, {st['rejected']} rejected", flush=True)
        if inject and chunks:
            inject.restore(chunks[-1][1])
        return [t for c, _ in chunks for t in c], finished

    def with_music(self, new: list[int], inject: MusicInjector | None) -> list[int]:
        """A window that filled up right after a SEG has no room for its music: add it before the next window."""
        seg = self.seg_of(new[-1]) if new and inject is not None else None
        return new if seg is None else [*new, *inject.music(seg)]

    def deco(self, header: list[int], objs: list[int]) -> list[int]:
        self.grammar.seg_cap = self.args.deco_seg_cap
        segs = [(i, s) for i, t in enumerate(objs) if (s := self.seg_of(t)) is not None]
        if not segs:
            return []
        first, last, w = segs[0][1], segs[-1][1], self.args.deco_window

        def layout_slice(lo: int, hi: int) -> list[int]:
            start = next((i for i, s in segs if s >= lo), None)
            if start is None:
                return []
            end = next((i for i, s in segs if s > hi), len(objs))
            return objs[start:end]

        out: list[int] = []
        prev: list[int] = []
        layout_segs = {s for _, s in segs}
        # decorate only around gameplay: a sample that jumps far ahead must not turn into thousands of empty windows
        windows = [s for s in range(first, last + 2, w) if any(x in layout_segs for x in range(s - 1, s + w + 2))]
        for s in windows[: self.args.max_deco_windows]:
            stop = {self.v["<EOS>"]} | {self.seg_token(x) for x in range(s + w, SEG_RANGE[1])}
            prompt = [self.v["<BOS>"], self.v["<DECO>"], *self.meta, *self.header_trunc(header), self.v["<GAMEPLAY>"],
                      *layout_slice(s - 1, s + w + 1)[: self.ctx // 2], self.v["<OBJECTS>"],
                      *([self.v["<CONT>"], *prev] if s > first else []), self.seg_token(s),
                      *self.music_ids(self.deco_music.get(s, []))]
            new = yield from self.sample(prompt, self.args.max_deco_window_tokens, stop,
                                         (lambda toks: self.music_ids(self.deco_music.get(self.seg_of(toks[-1]), [])))
                                         if self.deco_music else None)
            if new and new[-1] in stop:
                new = new[:-1]
            chunk = [self.seg_token(s), *self.music_ids(self.deco_music.get(s, [])), *new]
            out += chunk
            prev = self.prev_tail(chunk)
            if (s - first) // w % 10 == 0:
                print(f"{self.tag}  deco: segment {s}/{last}, {len(out)} tokens", flush=True)
        return out

    def prev_tail(self, chunk: list[int]) -> list[int]:
        """Last deco tokens as context for the next window: [SEG of the tail] + tail from an object start."""
        starts = [i for i in range(max(0, len(chunk) - self.args.deco_prev), len(chunk))
                  if self.grammar.cat[chunk[i]] == OBJ]
        if not starts:
            return []
        i = starts[0]
        seg = next((chunk[j] for j in range(i, -1, -1) if self.seg_of(chunk[j]) is not None), None)
        return ([seg] if seg is not None else []) + chunk[i:]


class ChunkStop:
    """`until` for GPT.generate: true at the first SEG k segments past the chunk's first SEG (that SEG is dropped)."""

    def __init__(self, gen: Generator, k: int):
        self.g, self.k, self.first, self.fired, self.goal_x = gen, k, None, False, None

    def __call__(self, toks: list[int]) -> bool:
        s = self.g.seg_of(toks[-1])
        if s is None:
            return False
        if self.first is None:
            self.first = s
            return False
        if s >= self.first + self.k:
            self.fired, self.goal_x = True, s * 300.0
        return self.fired


def load_model(args, device) -> tuple[GPT, Vocab]:
    vocab = Vocab.load(args.vocab)
    if args.random_init:
        cfg = GPTConfig.preset("test", vocab_size=128 * -(-len(vocab) // 128), ctx_len=1024)
        torch.manual_seed(args.seed)
        return GPT(cfg).to(device).eval(), vocab
    ck = torch.load(args.ckpt, map_location="cpu")          # the optimizer state in it never goes to the GPU
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    assert model.cfg.vocab_size >= len(vocab), "checkpoint and vocabulary do not match"
    return model.to(device).eval(), vocab


def publish_ready(args, name: str, header: dict, objects: list[dict]) -> None:
    """--ready-dir: the level string as soon as it exists (written whole, then renamed, so readers never see half)."""
    if not args.ready_dir:
        return
    d = Path(args.ready_dir)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{name}.part"
    tmp.write_text(encode_level_string(serialize_level(header, objects)))
    tmp.replace(d / f"{name}.txt")


def level_task(model: GPT, vocab: Vocab, args, seed: int, out: Path, ready: str, sims: ThreadPoolExecutor, tag=""):
    """One level from header to files, as a coroutine of sampling requests. Returns (report, simulation future)."""
    g = Generator(model, vocab, args, seed, tag)
    tok = Tokenizer(vocab)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    header, objs, finished = yield from g.layout()
    layout_toks = vocab.detokenize([vocab.index["<BOS>"], *header, vocab.index["<OBJECTS>"], *objs,
                                    vocab.index["<EOS>"]])
    head, gameplay = tok.decode_level(layout_toks, strict=False)
    layout_errors = tok.last_errors
    if g.features is not None:
        head["kA13"] = str(args.song_offset)                   # the offset the music tokens were computed with
        last = max((float(o.get("2", 0) or 0) for o in gameplay), default=0.0) // 300
        g.deco_music = level_music(head, gameplay, g.features, range(0, int(last) + args.deco_window + 2))
    publish_ready(args, f"{ready}-gameplay", head, gameplay)   # playable before the decoration is done
    print(f"{tag}  layout done: {len(objs)} tokens, {len(gameplay)} gameplay objects", flush=True)
    deco_ids = [] if args.no_deco else (yield from g.deco(header, objs))
    deco_toks = vocab.detokenize(deco_ids)
    _, deco_objs = tok.decode_level(["<OBJECTS>", *deco_toks, "<EOS>"], strict=False)
    deco_errors = tok.last_errors

    text = serialize_level(head, gameplay + deco_objs)
    (out / "level.txt").write_text(text)
    (out / "level_string.txt").write_text(encode_level_string(text))
    publish_ready(args, ready, head, gameplay + deco_objs)
    (out / "tokens_layout.txt").write_text(" ".join(layout_toks))
    (out / "tokens_deco.txt").write_text(" ".join(deco_toks))
    report = {"seed": seed, "layout_finished": finished, "layout_tokens": len(objs), "deco_tokens": len(deco_ids),
              "gameplay_objects": len(gameplay), "deco_objects": len(deco_objs), "layout_decode_errors": layout_errors,
              "deco_decode_errors": deco_errors, "seconds": round(time.time() - t0, 1),
              "selection": g.selection, "args": vars(args)}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{tag}  level done after {report['seconds']}s: {len(gameplay)} gameplay + {len(deco_objs)} deco objects",
          flush=True)
    # random tokens are not a level worth simulating; the search runs in a thread (ctypes releases the GIL)
    future = sims.submit(simulate, head, gameplay, args.max_frontier, args.sim_seconds) \
        if gameplay and not args.random_init else None
    return report, future


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--vocab", default=str(VOCAB_FILE))
    ap.add_argument("--out", required=True)
    ap.add_argument("--stars", type=int, default=8)
    ap.add_argument("--demon", type=int, default=0)
    ap.add_argument("--epic", type=int, default=1)
    ap.add_argument("--length", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0, help="below 1 the model falls into copy loops sooner")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1, help="levels, seeds seed..seed+samples-1 (into <out>/s<seed>)")
    ap.add_argument("--batch", type=int, default=8, help="sampling requests decoded together")
    ap.add_argument("--max-layout-tokens", type=int, default=60_000)
    ap.add_argument("--deco-window", type=int, default=2)
    ap.add_argument("--max-deco-window-tokens", type=int, default=8000)
    ap.add_argument("--max-deco-windows", type=int, default=1000)
    ap.add_argument("--deco-prev", type=int, default=512)
    ap.add_argument("--header-max", type=int, default=768)
    ap.add_argument("--no-deco", action="store_true")
    ap.add_argument("--max-seg-skip", type=int, default=3, help="empty segments in a row at most (-1: any)")
    ap.add_argument("--max-same-x", type=int, default=16, help="objects in a row at one x at most (0: any)")
    ap.add_argument("--layout-seg-cap", type=int, default=3000, help="layout tokens per segment at most")
    ap.add_argument("--deco-seg-cap", type=int, default=12000, help="decoration tokens per segment at most")
    ap.add_argument("--song-features", help="song features .npz (scripts/song_features.py): build the level to it")
    ap.add_argument("--song-offset", type=float, default=0.0, help="seconds into the song where the level starts")
    ap.add_argument("--select-segments", type=int, default=0, help="check the layout every K segments (0: off)")
    ap.add_argument("--select-tries", type=int, default=4, help="samples of a chunk before giving up on it")
    ap.add_argument("--select-backtracks", type=int, default=4, help="previous chunks redone per level at most")
    ap.add_argument("--select-sim-seconds", type=float, default=20, help="simulator budget per check")
    ap.add_argument("--ready-dir", help="also write <name>-gameplay.txt after the layout and <name>.txt at the end here")
    ap.add_argument("--ready-name", default="level")
    ap.add_argument("--max-frontier", type=int, default=256)
    ap.add_argument("--sim-seconds", type=float, default=300, help="time budget of the layout simulation")
    args = ap.parse_args()
    if not args.ckpt and not args.random_init:
        ap.error("--ckpt or --random-init")

    if args.max_seg_skip < 0:
        args.max_seg_skip = None
    args.max_same_x = args.max_same_x or None
    torch.set_num_threads(2)              # the CPU work is small per-token mask ops: many threads only contend
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = load_model(args, device)
    out = Path(args.out)
    seeds = list(range(args.seed, args.seed + args.samples))
    with ThreadPoolExecutor(max_workers=4) as sims:
        if len(seeds) == 1:
            results = [run_sync(level_task(model, vocab, args, seeds[0], out, args.ready_name, sims), model)]
            dirs = [out]
        else:
            dirs = [out / f"s{s}" for s in seeds]
            tasks = [level_task(model, vocab, args, s, d, f"ready-s{s}", sims, tag=f"[s{s}]") for s, d in zip(seeds, dirs)]
            results = BatchDecoder(model, rows=min(args.batch, len(seeds))).run(tasks)
        for d, (report, future) in zip(dirs, results):
            try:
                report["simulation"] = future.result() if future is not None else None
            except Exception as e:  # noqa: BLE001  (a malformed sample must not lose the files written above)
                report["simulation"] = {"error": f"{type(e).__name__}: {e}"}
            (d / "report.json").write_text(json.dumps(report, indent=1))
            print(json.dumps({k: v for k, v in report.items() if k != "args"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
