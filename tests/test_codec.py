from pathlib import Path

from gdgen.levelstring import (
    decode_level_string,
    decompress,
    encode_level_string,
    parse_level,
    serialize_level,
    to_storage_bytes,
)
from gdgen.server import parse_download_response, parse_search_response

FIXTURES = Path(__file__).parent / "fixtures"

LEVEL_TEXT = "kS38,1_40_2_125_3_255,kA2,0;1,1,2,15,3,15;1,8,2,45,3,15,6,90;1,901,2,75,3,105,57,1.2;"


def test_search_page_fixture():
    page = parse_search_response((FIXTURES / "search_page.txt").read_text())
    assert len(page.levels) == 10
    assert page.amount == 10 and page.offset == 0 and page.total > 30000
    assert all(lvl["1"].isdigit() for lvl in page.levels)
    assert len(page.creators) >= 1
    assert page.songs and "1" in page.songs[0]


def test_empty_responses():
    assert parse_search_response("-1").levels == []
    assert parse_download_response("-1") is None


def test_download_response():
    encoded = encode_level_string(LEVEL_TEXT)
    meta = parse_download_response(f"1:123:2:Example:4:{encoded}:45:3#hash1#hash2")
    assert meta["1"] == "123" and meta["2"] == "Example"
    assert decode_level_string(meta["4"]) == LEVEL_TEXT


def test_codec_roundtrip():
    blob = to_storage_bytes(encode_level_string(LEVEL_TEXT))
    assert decompress(blob) == LEVEL_TEXT


def test_plain_level_string_is_accepted():
    assert decompress(to_storage_bytes(LEVEL_TEXT)) == LEVEL_TEXT


def test_parse_and_serialize():
    header, objects = parse_level(LEVEL_TEXT)
    assert header["kA2"] == "0"
    assert [o["1"] for o in objects] == ["1", "8", "901"]
    assert objects[2]["57"] == "1.2"
    assert serialize_level(header, objects) == LEVEL_TEXT
