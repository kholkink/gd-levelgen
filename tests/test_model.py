"""Model and training windows: KV-cache consistency, loss masking, window layout on a tiny pretokenized dataset."""
import json

import numpy as np
import pytest
import torch

from gdgen.model.data import IGNORE, TokenData, WindowSampler
from gdgen.model.gpt import GPT, GPTConfig
from gdgen.tokens import KindCounter, PairCounter, Tokenizer, build_vocab


def tiny_model(vocab_size=97, ctx=64) -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig.preset("test", vocab_size=vocab_size, ctx_len=ctx, loss_chunk=16))


def test_kv_cache_matches_full_forward():
    model = tiny_model().eval()
    seq = torch.randint(0, 97, (1, 40))
    full = model(seq)[0]
    caches = [{} for _ in model.blocks]
    h = model.hidden(seq[:, :25], 0, caches)                    # prefill
    got = [model.logits(h)[0]]
    h = model.hidden(seq[:, 25:31], 25, caches)                 # chunk on top of the cache
    got.append(model.logits(h)[0])
    for t in range(31, 40):                                     # one token at a time
        got.append(model.logits(model.hidden(seq[:, t:t + 1], t, caches))[0])
    assert torch.allclose(torch.cat(got), full, atol=1e-4)


def test_loss_ignores_masked_targets_and_trains():
    model = tiny_model()
    x = torch.randint(0, 97, (2, 48))
    y = x.roll(-1, dims=1)
    y[:, :10] = IGNORE
    loss, cnt = model(x, y, grad_ckpt=True)
    assert int(cnt) == 2 * 38
    logits = model(x)
    ref = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=IGNORE, reduction="sum")
    assert torch.allclose(loss, ref, rtol=1e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(30):                                         # memorizes one batch
        loss, cnt = model(x, y)
        (loss / cnt).backward()
        opt.step()
        opt.zero_grad()
        first = first or (loss / cnt).item()
    assert (loss / cnt).item() < 0.5 * first


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """One level, written exactly as scripts/pretokenize.py writes it."""
    gameplay = [{"1": "1", "2": str(15 + 30 * i), "3": "15", "21": "1004"} for i in range(60)]
    deco = [{"1": "211", "2": str(20 + 13 * i), "3": str(60 + i % 5), "21": "2", "24": "5"} for i in range(140)]
    header = {"kA2": "0", "kA4": "0", "kS38": "1_0_2_0_3_0_6_1000|1_40_2_40_3_40_6_1|"}
    kc = KindCounter()
    kc.add_header(header)
    for o in gameplay + deco:
        kc.add_object(o)
    pc = PairCounter(kc.kinds())
    for o in gameplay + deco:
        pc.add_object(o)
    vocab = build_vocab(kc, pc, min_count=1)
    tok = Tokenizer(vocab)
    layout = vocab.ids(tok.encode_level(header, gameplay))
    deco_ids = vocab.ids(["<OBJECTS>", *tok.encode_objects(deco), "<EOS>"])
    root = tmp_path_factory.mktemp("tokens")
    vocab.save(root / "vocab.json")
    np.asarray(layout, dtype=np.uint16).tofile(root / "layout_000.bin")
    np.asarray(deco_ids, dtype=np.uint16).tofile(root / "deco_000.bin")
    rec = {"id": 1, "shard": 0, "layout": [0, len(layout)], "deco": [0, len(deco_ids)],
           "header_len": tok.encode_level(header, gameplay).index("<OBJECTS>") + 1, "stars": 8, "demon": 0,
           "epic": 1, "length": 3, "game_version": 22, "unk": 0}
    (root / "index.jsonl").write_text(json.dumps(rec) + "\n")
    return root, vocab


def tokens_of(vocab, ids):
    return [vocab.tokens[i] for i in ids]


def test_layout_windows(dataset):
    root, vocab = dataset
    sampler = WindowSampler(TokenData(root, "train", val_every=10**9), ctx_len=128, p_fresh=0.0, seed=1)
    for _ in range(20):
        x, y, kind = sampler.sample("layout")
        toks = tokens_of(vocab, x)
        assert kind == "layout" and toks[:2] == ["<BOS>", "<LAYOUT>"] and "M:stars=8" in toks[:6]
        cont = toks.index("<CONT>")
        assert toks[cont + 1].startswith("SEG:")
        assert (y[:cont] == IGNORE).all() and y[cont] != IGNORE      # first prediction is the SEG after <CONT>
        assert all(t.startswith(("SEG:", "O:", "X:", "Y:", "YF:", "A:", "<EOS>", "<PAD>")) for t in toks[cont + 1:])


def test_deco_windows(dataset):
    root, vocab = dataset
    sampler = WindowSampler(TokenData(root, "train", val_every=10**9), ctx_len=256, deco_prev=40, seed=2)
    seen_cont = False
    for _ in range(30):
        x, y, kind = sampler.sample("deco")
        assert kind == "deco"
        toks = tokens_of(vocab, x)
        g, o = toks.index("<GAMEPLAY>"), toks.index("<OBJECTS>")
        assert toks[1] == "<DECO>" and g < o
        assert all(t != "O:211" for t in toks[g:o])                  # gameplay context holds no deco objects
        first = int(np.flatnonzero(y != IGNORE)[0]) + 1             # sequence index of the first predicted token
        assert first > o and toks[first].startswith("SEG:")
        if "<CONT>" in toks:
            seen_cont = True
            prev = toks[o + 2:first]
            assert prev and prev[0].startswith("SEG:") and prev[1] == "O:211"   # tail starts on an object
            assert int(prev[0][4:]) < int(toks[first][4:])
    assert seen_cont


def test_grammar_masks(dataset):
    from gdgen.model.grammar import Grammar
    _, vocab = dataset
    g, v = Grammar(vocab, size=len(vocab) + 5), vocab.index         # model vocabulary padded by 5 ids
    allowed = lambda m: {vocab.tokens[i] for i in torch.nonzero(m[: len(vocab)]).flatten().tolist()}  # noqa: E731
    toks = [v["<BOS>"], v["<LAYOUT>"], v["M:stars=8"]]
    pre = allowed(g.allowed(toks))
    assert "<HEADER>" in pre and not any(t.startswith(("O:", "V:", "H:")) for t in pre)
    assert not g.allowed(toks)[len(vocab):].any()
    toks += [v["<HEADER>"]]
    assert all(t.startswith("H:") for t in allowed(g.allowed(toks)))
    toks += [v["H:kA4"], v["V:0"]]
    after = allowed(g.allowed(toks))
    assert "H:kA2" not in after and "H:kA13" in after and "<COLORS>" in after and "V:1" not in after  # one value
    toks += [v["<COLORS>"], v["<CHAN>"], v["CP:6"], v["V:1000"], v["<OBJECTS>"], v["SEG:3"], v["O:1"]]
    assert all(t.startswith("X:") for t in allowed(g.allowed(toks)))
    toks += [v["X:4"], v["Y:0"], v["YF:60"]]
    after = allowed(g.allowed(toks))
    assert "SEG:4" in after and "SEG:3" not in after and "SEG:-8" not in after   # segments only move forward
    assert "O:1" in after and "<EOS>" in after and "X:4" not in after and "V:5" not in after
    assert "O:211" not in after                                      # a <LAYOUT> stream holds gameplay objects only
    toks += [v["A:21=1004"]]                                         # attributes only in increasing key order
    after = allowed(g.allowed(toks))
    assert "A:24=5" in after and "A:21=1004" not in after and not any(t.startswith(("F:", "G:")) for t in after)
    toks += [v["K:24"]]
    assert all(t.startswith(("V:", "D:", "B:", "H", "TRAW", "EMPTY", "DEND", "LEND")) for t in allowed(g.allowed(toks)))


def test_tolerant_decode(dataset):
    _, vocab = dataset
    tok = Tokenizer(vocab)
    good = ["SEG:0", "O:1", "X:60", "Y:0", "YF:60"]
    broken = ["O:1", "X:64", "YF:3", "V:5"]                          # Y missing
    stream = ["<OBJECTS>", *good, *broken, "SEG:1", *good[1:], "<EOS>"]
    with pytest.raises(ValueError):
        tok.decode_level(stream)
    _, objs = tok.decode_level(stream, strict=False)
    assert tok.last_errors == 1 and [o["2"] for o in objs] == ["15", "315"]


def test_grammar_accepts_every_encoder_stream():
    """Every token the encoder writes must be allowed by the mask computed from the tokens before it."""
    from test_tokens import FIT_OBJECTS, HEADER

    from gdgen.model.grammar import Grammar
    from gdgen.objects import GAMEPLAY_CATEGORIES, category_of

    extras = [
        {"1": "1", "2": "10", "3": "20", "21": "777", "43": "-15a0.75a-0.3a0a1", "128": "3.3", "129": "3.3"},
        {"1": "1", "2": "11", "3": "20", "64": "2", "57": "3.12000", "51": "abc"},
        {"1": "99999", "2": "12", "3": "20"},
        {"1": "914", "2": "13", "3": "20", "31": "", "10": "123456.789"},
    ]
    objects = FIT_OBJECTS + extras
    kc = KindCounter()
    kc.add_header(HEADER)
    for o in objects:
        kc.add_object(o)
    pc = PairCounter(kc.kinds())
    for o in FIT_OBJECTS:
        pc.add_object(o)
    vocab = build_vocab(kc, pc, min_count=1)
    tok, v = Tokenizer(vocab), vocab.index
    gameplay = [o for o in objects if o["1"] == "99999" or category_of(int(o["1"])) in GAMEPLAY_CATEGORIES]
    streams = {
        "plain": vocab.ids(tok.encode_level(HEADER, objects)),
        "layout": [v["<BOS>"], v["<LAYOUT>"], v["M:stars=8"], *vocab.ids(tok.encode_level(HEADER, gameplay))[1:]],
    }
    for name, seq in streams.items():
        g = Grammar(vocab, size=len(vocab) + 3)
        for i in range(1, len(seq)):
            ok = bool(g.allowed(seq[:i])[seq[i]])
            assert ok, f"{name}: {vocab.tokens[seq[i]]} rejected after {vocab.detokenize(seq[max(0, i - 8):i])}"
