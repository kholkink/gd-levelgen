"""Tokenizer round trip: decode(encode(level)) == canonical(level), on a small vocabulary fitted in the test."""
import base64

import pytest

from gdgen.tokens import (
    VOCAB_FILE,
    KindCounter,
    PairCounter,
    Tokenizer,
    Vocab,
    build_vocab,
    canonical_header,
    canonical_object,
    norm_angle,
    norm_num,
    norm_scale,
)

HEADER = {
    "kS38": "1_0_2_102_3_255_6_1000_7_1_15_1_18_0_8_1|1_40_2_40_3_40_6_1_7_0.5_9_1000_10_20a0.8a1a0a0_17_1|",
    "kA2": "0", "kA4": "1", "kA13": "203.5", "kA14": "211.95~0~212.42~0", "kA36": "261",
}
TEXT = base64.urlsafe_b64encode(b"hello GD").decode()
FIT_OBJECTS = [
    {"1": "1", "2": "15", "3": "15", "64": "1", "67": "1", "96": "1", "21": "1004", "24": "5", "155": "2"},
    {"1": "1", "2": "45", "3": "15", "64": "1", "67": "1", "96": "1", "21": "1004", "24": "5", "20": "3"},
    {"1": "8", "2": "75", "3": "15.25", "6": "-90", "57": "5.12", "128": "2", "129": "2", "4": "1"},
    {"1": "211", "2": "1234.5678", "3": "99.9", "43": "0a1a0.5a0a0", "41": "1", "128": "0.733", "129": "1.875"},
    {"1": "211", "2": "-30", "3": "45", "43": "0a1a0.5a0a0", "41": "1", "131": "33.333", "132": "-12.5"},
    {"1": "901", "2": "300", "3": "1665", "51": "9", "28": "-30", "29": "0", "10": "0.5", "30": "3", "85": "2",
     "36": "1", "13": "0"},
    {"1": "914", "2": "600", "3": "105", "31": TEXT},
    {"1": "2065", "2": "615", "3": "105", "145": "30a-1a1a0.3a30a90a90a29a0a11a0a0a0a0a0a0a0a1"},
    {"1": "1268", "2": "630", "3": "105", "442": "1.2.3.4", "63": "0.25"},
]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    kc = KindCounter()
    kc.add_header(HEADER)
    for o in FIT_OBJECTS:
        kc.add_object(o)
    pc = PairCounter(kc.kinds())
    for o in FIT_OBJECTS:
        pc.add_object(o)
    return Tokenizer(build_vocab(kc, pc, min_count=1))


def roundtrip(tok: Tokenizer, header, objects):
    tokens = tok.encode_level(header, objects)
    dh, dobjs = tok.decode_level(tokens)
    canon = [canonical_object(o, tok.kinds) for o in objects]
    key = lambda o: (float(o["2"]), float(o["3"]))  # noqa: E731
    assert dh == canonical_header(header)
    assert sorted(dobjs, key=key) == sorted(canon, key=key)
    return tokens, dobjs


def test_number_normalization():
    assert norm_num("-90.000000") == "-90"
    assert norm_num("0.12345") == "0.123"
    assert norm_num("-0.0001") == "0"
    assert norm_num("abc") is None
    assert norm_angle("-90") == "270" and norm_angle("720") == "0" and norm_angle("179.79") == "179.8"
    assert norm_scale("1.875") == "1.875" and norm_scale("0.733") == "0.735" and norm_scale("7.33") == "7.35"


def test_fitted_kinds(tok):
    kinds = tok.kinds
    assert kinds["57"] == "groups" and kinds["43"] == "hsv" and kinds["31"] == "text"
    assert kinds["145"] == "astr" and kinds["442"] == "list"
    assert kinds["64"] == "flag" and kinds["13"] == "flag" and kinds["21"] == "num"
    assert "155" not in kinds and "20" not in kinds            # editor-only keys are not modeled


def test_fit_objects_roundtrip(tok):
    tokens, _ = roundtrip(tok, HEADER, FIT_OBJECTS)
    assert tokens[0] == "<BOS>" and tokens[-1] == "<EOS>"
    assert "<UNK>" not in tokens
    assert "kA14" not in tok.decode_level(tokens)[0]           # editor guidelines dropped


def test_fused_tokens(tok):
    _, toks = tok.encode_object(FIT_OBJECTS[0])
    assert toks[:4] == ["O:1", "X:60", "Y:0", "YF:60"]
    assert "FS:64.67.96" in toks and "A:21=1004" in toks and "A:24=5" in toks
    _, toks = tok.encode_object(FIT_OBJECTS[2])
    assert "A:6=270" in toks and "A:s=2" in toks and toks.count("G:5") == 1 and "G:12" in toks
    _, toks = tok.encode_object(FIT_OBJECTS[5])
    assert "F0:13" in toks


def test_fallbacks_roundtrip(tok):
    objects = [
        # unseen pair values: digits for numbers, compact HSV, separate scale keys
        {"1": "1", "2": "10", "3": "20", "21": "777", "43": "-15a0.75a-0.3a0a1", "128": "3.3", "129": "3.3"},
        # a flag key holding a non-flag value, a group list with an out-of-range id, a non-numeric value
        {"1": "1", "2": "11", "3": "20", "64": "2", "57": "3.12000", "51": "abc"},
        # a key never seen in the fit is dropped, an unknown object id survives
        {"1": "99999", "2": "12", "3": "20", "999": "5"},
        # empty text and a value only representable with digits
        {"1": "914", "2": "13", "3": "20", "31": "", "10": "123456.789"},
    ]
    tokens, dobjs = roundtrip(tok, HEADER, objects)
    assert "<UNK>" not in tokens
    assert "999" not in dobjs[2] and dobjs[2]["1"] == "99999"


def test_segments_and_order(tok):
    objects = [{"1": "1", "2": str(x), "3": "15"} for x in (905, -20, 15, 299.9, 300)]
    tokens = tok.encode_objects(objects)
    segs = [t for t in tokens if t.startswith("SEG:")]
    assert segs == ["SEG:-1", "SEG:0", "SEG:1", "SEG:3"]
    _, dobjs = tok.decode_level(["<OBJECTS>", *tokens])
    assert [o["2"] for o in dobjs] == ["-20", "15", "300", "300", "905"]   # 299.9 rounds to 300


@pytest.mark.skipif(not VOCAB_FILE.exists(), reason="fitted vocabulary not built")
def test_shipped_vocab_fits_uint16():
    vocab = Vocab.load()
    assert len(vocab) <= 65535
    assert len(set(vocab.tokens)) == len(vocab.tokens)
