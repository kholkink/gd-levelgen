"""Music conditioning: level clock, per-segment music tokens, tokenizer round trip with music, forced injection."""
import numpy as np
import pytest
import torch

from gdgen.model.gpt import GPT, GPTConfig
from gdgen.model.grammar import Grammar
from gdgen.music import SEG_UNITS, LevelClock, MusicInjector, level_music, x_speed
from gdgen.tokens import MUSIC_TOKENS, KindCounter, PairCounter, Tokenizer, build_vocab, canonical_object


def features(seconds=20.0, fps=43.0, bpm=120.0):
    n = int(seconds * fps)
    t = np.arange(n) / fps
    loud = np.where(t < 10, -30.0, -3.0)                  # quiet first half, loud second half
    return {"rms": loud.astype(np.float16), "onset": np.full(n, 0.5, np.float16),
            "low": loud.astype(np.float16), "beats": np.arange(0, seconds, 60 / bpm, dtype=np.float32),
            "fps": np.float32(fps), "tempo": np.float32(bpm), "duration": np.float32(seconds)}


def test_clock_follows_speed_portals():
    header = {"kA4": "0"}                                   # normal speed 0.9
    objs = [{"1": "202", "2": "600", "3": "15"}]            # 2x speed portal (1.1) at x = 600
    clock = LevelClock(header, objs)
    assert clock.time(600) == pytest.approx(600 / x_speed(0.9))
    assert clock.time(900) == pytest.approx(600 / x_speed(0.9) + 300 / x_speed(1.1))
    assert clock.x(clock.time(1234.5)) == pytest.approx(1234.5)


def test_segment_tokens_track_loudness_and_beats():
    feats = features()
    objs = [{"1": "1", "2": str(x), "3": "15"} for x in range(15, 6000, 300)]
    music = level_music({"kA4": "0", "kA13": "0"}, objs, feats)
    seg_seconds = SEG_UNITS / x_speed(0.9)                  # ~0.87 s per segment
    quiet, loud = music[1], music[int(12 / seg_seconds)]
    assert quiet[0] == "MU:e=4" and loud[0] == "MU:e=7"
    beats = [t for t in music[3] if t.startswith("MU:b=")]
    assert 1 <= len(beats) <= 2                              # 2 beats per second over ~0.87 s
    offset = level_music({"kA4": "0", "kA13": "10"}, objs, feats)   # song starts 10 s in: loud from the start
    assert offset[1][0] == "MU:e=7"


@pytest.fixture(scope="module")
def tok():
    objs = [{"1": "1", "2": "15", "3": "15", "21": "1004"}, {"1": "8", "2": "945", "3": "15"}]
    kc = KindCounter()
    for o in objs:
        kc.add_object(o)
    pc = PairCounter(kc.kinds())
    for o in objs:
        pc.add_object(o)
    return Tokenizer(build_vocab(kc, pc, min_count=1)), objs


def test_music_tokens_roundtrip(tok):
    tokenizer, objs = tok
    assert all(t in tokenizer.vocab for t in MUSIC_TOKENS)
    music = {0: ["MU:e=3", "MU:o=1", "MU:l=2", "MU:b=4"], 1: ["MU:e=5", "MU:o=0", "MU:l=0"],
             3: ["MU:e=7", "MU:o=7", "MU:l=7", "MU:b=0", "MU:b=9"], 4: ["MU:e=1", "MU:o=1", "MU:l=1"]}
    tokens = tokenizer.encode_objects(objs, music)
    assert tokens[:5] == ["SEG:0", "MU:e=3", "MU:o=1", "MU:l=2", "MU:b=4"]
    assert "SEG:1" in tokens and tokens[tokens.index("SEG:1") + 1] == "MU:e=5"          # music-only segment
    assert [t for t in tokens if t.startswith("SEG:")] == ["SEG:0", "SEG:1", "SEG:3", "SEG:4"]
    _, decoded = tokenizer.decode_level(["<OBJECTS>", *tokens, "<EOS>"])
    assert decoded == [canonical_object(o, tokenizer.kinds) for o in objs]


def test_generate_injects_music_and_grammar_keeps_segments_consecutive(tok):
    tokenizer, _ = tok
    vocab = tokenizer.vocab
    v = vocab.index
    torch.manual_seed(0)
    model = GPT(GPTConfig.preset("test", vocab_size=len(vocab), ctx_len=256)).eval()
    grammar = Grammar(vocab, len(vocab), consecutive=True)
    seg_lo = v["SEG:-8"]

    def inject(toks):
        s = toks[-1] - seg_lo
        return [v["MU:e=2"], v["MU:o=2"], v["MU:l=2"]] if 0 <= s < 2048 else []

    prompt = [v["<BOS>"], v["<LAYOUT>"], v["M:stars=8"], v["<HEADER>"], v["<COLORS>"], v["<OBJECTS>"]]
    out = model.generate(prompt, 120, 1.0, 1.0, allowed=grammar.allowed, inject=inject,
                         generator=torch.Generator().manual_seed(1))
    names = vocab.detokenize(out)
    segs = [int(t[4:]) for t in names if t.startswith("SEG:")]
    assert segs and all(b == a + 1 for a, b in zip(segs, segs[1:]))                   # no skipped segment
    for i, t in enumerate(names):
        if t.startswith("SEG:") and i + 3 < len(names):
            assert names[i + 1: i + 4] == ["MU:e=2", "MU:o=2", "MU:l=2"]               # music right after each SEG


def test_injector_reproduces_training_music():
    """Walking an encoded training stream token by token, the injector forces exactly the music that follows each SEG
    (the clock of a segment only uses the portals before it, which is what has been sampled at its SEG)."""
    objs = [{"1": "1", "2": str(x), "3": "15"} for x in range(15, 4000, 210)]
    objs += [{"1": "202", "2": "705", "3": "45"}, {"1": "200", "2": "2115", "3": "45"},
             {"1": "203", "2": "2400", "3": "45", "121": "1"}]                  # 2x, 0.5x, a no-touch 3x
    kc = KindCounter()
    for o in objs:
        kc.add_object(o)
    pc = PairCounter(kc.kinds())
    for o in objs:
        pc.add_object(o)
    tokenizer = Tokenizer(build_vocab(kc, pc, min_count=1))
    header, feats = {"kA4": "0", "kA13": "1.5"}, features(seconds=60.0)
    music = level_music(header, objs, feats)
    ids = tokenizer.vocab.ids(["<OBJECTS>", *tokenizer.encode_objects(objs, music), "<EOS>"])
    names = tokenizer.vocab.detokenize(ids)
    inj = MusicInjector(tokenizer, feats, header, 1.5)
    checked = 0
    for i in range(1, len(ids)):
        if names[i].startswith("MU:"):
            continue
        forced = inj(ids[: i + 1])
        following = names[i + 1: i + 2 + len(forced)]
        expect = []
        for t in following:
            if not t.startswith("MU:"):
                break
            expect.append(t)
        assert tokenizer.vocab.detokenize(forced) == expect, (names[i], expect)
        checked += bool(forced)
    assert checked == len(music) > 10
