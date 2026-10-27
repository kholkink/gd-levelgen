"""Level string codec: compressed transport form <-> raw text <-> header + object dicts.

Raw level text is `header;obj;obj;...`, where each object is `key,value,key,value,...`
(key 1 = object id, 2 = x, 3 = y). Positions are in units where one grid cell is 30.
"""
from __future__ import annotations

import base64
import gzip
import zlib


def _b64decode(s: str) -> bytes:
    s = s.strip()
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def to_storage_bytes(level_string: str) -> bytes:
    """Server level string -> compressed bytes (gzip or zlib) suitable for a BLOB."""
    try:
        blob = _b64decode(level_string)
        zlib.decompress(blob, 15 | 32)  # validate
        return blob
    except (ValueError, zlib.error):
        pass
    # Very old levels are stored uncompressed.
    if level_string.startswith(("kS", "kA", "1,")):
        return gzip.compress(level_string.encode("utf-8"), compresslevel=6)
    raise ValueError(f"unrecognized level string: {level_string[:40]!r}")


def decompress(blob: bytes) -> str:
    return zlib.decompress(blob, 15 | 32).decode("utf-8", "replace")


def decode_level_string(level_string: str) -> str:
    return decompress(to_storage_bytes(level_string))


def level_text(s: str) -> str:
    """Raw level text from raw text itself or from a base64 level string (which never contains ';' or ',')."""
    s = s.strip()
    return s if ";" in s or "," in s else decode_level_string(s)


def encode_level_string(text: str) -> str:
    """Raw level text -> the base64(gzip) form the game and servers use."""
    return base64.urlsafe_b64encode(gzip.compress(text.encode("utf-8"))).decode("ascii")


def parse_object(obj: str) -> dict[str, str]:
    kv = obj.split(",")
    return dict(zip(kv[::2], kv[1::2]))


def parse_header(header: str) -> dict[str, str]:
    return parse_object(header)


def parse_level(text: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    parts = text.split(";")
    header = parse_header(parts[0])
    objects = [parse_object(p) for p in parts[1:] if p]
    return header, objects


def serialize_object(obj: dict[str, str]) -> str:
    return ",".join(f"{k},{v}" for k, v in obj.items())


def serialize_level(header: dict[str, str], objects: list[dict[str, str]]) -> str:
    return ";".join([serialize_object(header), *(serialize_object(o) for o in objects)]) + ";"
