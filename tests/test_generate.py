"""Simulator-guided layout selection of scripts/generate.py with a random tiny model and a scripted checker."""
import argparse
import importlib.util
from pathlib import Path

import pytest

from gdgen.model.gpt import run_sync
from gdgen.tokens import VOCAB_FILE

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("generate_script", ROOT / "scripts" / "generate.py")
generate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate)


def generator(**over):
    args = dict(random_init=True, ckpt=None, vocab=str(VOCAB_FILE), seed=0, stars=8, demon=0, epic=1, length=3,
                temperature=1.0, top_p=1.0, max_layout_tokens=240, header_max=64, song_features=None, song_offset=0.0,
                select_segments=2, select_tries=3, select_backtracks=1, select_sim_seconds=1.0, max_frontier=16,
                max_seg_skip=None, layout_seg_cap=None, deco_seg_cap=None, max_same_x=None)
    args.update(over)
    args = argparse.Namespace(**args)
    model, vocab = generate.load_model(args, "cpu")
    g = generate.Generator(model, vocab, args)
    return g, [vocab.index["<HEADER>"], vocab.index["<COLORS>"]]


@pytest.fixture(scope="module")
def passing_every_third():
    g, header = generator()
    calls = []

    def check(header, objs, goal_x):
        calls.append((list(objs), goal_x))
        return {"passed": len(calls) % 3 == 0, "max_x": float(len(calls)), "seconds": 0.0}

    g.check = check
    objs, _ = run_sync(g.layout_selected(header, None), g.model)
    return g, objs, calls


def test_rejected_chunks_are_resampled(passing_every_third):
    g, objs, calls = passing_every_third
    st = g.selection
    assert st["attempts"] == len(calls) and st["chunks"] >= 2
    assert st["rejected"] == 2 * st["chunks"] and st["backtracks"] == 0 and st["unresolved_at"] is None
    assert calls[-1][0] == objs                          # the level is exactly the accepted (passing) tries


def test_chunks_stop_at_a_later_segment(passing_every_third):
    g, objs, calls = passing_every_third
    goals = [goal for _, goal in calls if goal is not None]
    assert goals and all(goal % 300 == 0 for goal in goals)
    for prefix, goal in calls:                           # nothing at or past the segment the chunk stopped at
        if goal is not None:
            segs = [s for t in prefix if (s := g.seg_of(t)) is not None]
            assert max(segs) * 300 < goal


def test_backtrack_then_give_up():
    g, header = generator(max_layout_tokens=160)
    calls = []

    def check(header, objs, goal_x):
        calls.append(len(objs))
        return {"passed": len(calls) == 1, "max_x": 10.0, "seconds": 0.0}

    g.check = check
    run_sync(g.layout_selected(header, None), g.model)
    st = g.selection
    assert st["backtracks"] == 1 and st["unresolved_at"] == 10.0
    assert len(calls) == 1 + 3 + 3                       # chunk 1 ok, chunk 2 fails, chunk 1 again fails, unchecked


def test_grammar_caps_segment_skips_and_length():
    from gdgen.model.grammar import Grammar
    from gdgen.tokens import Vocab

    vocab = Vocab.load(VOCAB_FILE)
    v = vocab.index
    g = Grammar(vocab, len(vocab), max_skip=1, seg_cap=5)
    toks = [v[t] for t in ["<BOS>", "<LAYOUT>", "M:stars=8", "<HEADER>", "<COLORS>", "<OBJECTS>",
                           "SEG:0", "O:1", "X:60", "Y:0", "YF:60"]]
    m = g.allowed(toks)
    assert m[v["O:1"]] and m[v["SEG:1"]] and m[v["SEG:2"]] and not m[v["SEG:3"]]   # one empty segment at most
    toks += [v["O:1"], v["X:64"], v["Y:0"], v["YF:60"]]                             # 8 tokens since SEG:0 > 5
    m = g.allowed(toks)
    assert not m[v["O:1"]] and m[v["SEG:1"]] and m[v["<EOS>"]]
    toks += [v["SEG:1"], v["O:1"]]
    assert g.allowed(toks)[v["X:60"]]                                               # the count restarts


def test_grammar_positions_never_go_back_inside_a_segment():
    from gdgen.model.grammar import Grammar
    from gdgen.tokens import Vocab

    vocab = Vocab.load(VOCAB_FILE)
    v = vocab.index
    g = Grammar(vocab, len(vocab))
    toks = [v[t] for t in ["<BOS>", "<LAYOUT>", "M:stars=8", "<HEADER>", "<COLORS>", "<OBJECTS>",
                           "SEG:0", "O:1", "X:500", "Y:3", "YF:60", "O:1"]]
    m = g.allowed(toks)
    assert not m[v["X:400"]] and m[v["X:500"]] and m[v["X:900"]]
    m = g.allowed(toks + [v["X:500"]])
    assert not m[v["Y:2"]] and m[v["Y:3"]] and m[v["Y:9"]]                     # same x: y does not go down
    assert g.allowed(toks + [v["X:500"], v["Y:3"]])[v["YF:60"]]
    assert not g.allowed(toks + [v["X:500"], v["Y:3"]])[v["YF:59"]]
    assert g.allowed(toks + [v["X:501"]])[v["Y:0"]]                            # larger x: any y
    g.reset()
    m = g.allowed(toks[:-1] + [v["SEG:1"], v["O:1"]])
    assert m[v["X:0"]]                                                          # a new segment starts over


def test_grammar_stops_towers_at_one_x():
    from gdgen.model.grammar import Grammar
    from gdgen.tokens import Vocab

    vocab = Vocab.load(VOCAB_FILE)
    v = vocab.index
    g = Grammar(vocab, len(vocab), max_same_x=3)
    toks = [v[t] for t in ["<BOS>", "<LAYOUT>", "M:stars=8", "<HEADER>", "<COLORS>", "<OBJECTS>", "SEG:0"]]
    for y in (1, 2):
        toks += [v["O:1"], v["X:500"], v[f"Y:{y}"], v["YF:0"]]
    assert g.allowed(toks + [v["O:1"]])[v["X:500"]]                    # two in a row: a third is fine
    toks += [v["O:1"], v["X:500"], v["Y:3"], v["YF:0"], v["O:1"]]
    m = g.allowed(toks)
    assert not m[v["X:500"]] and m[v["X:501"]]                          # three: move right
