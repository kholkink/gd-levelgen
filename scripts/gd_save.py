"""Adds generated levels to Geometry Dash's own save (Create -> My Levels), no mod needed.

The save CCLocalLevels.dat is a plist (gzip, base64, every byte XOR 11); its LLM_01 array holds the local levels,
newest first. Every run backs both save files up to replay/save_backup/<time>/ first, and refuses to write while
GeometryDash.exe runs (the game rewrites the save from memory when it closes).

    python scripts/gd_save.py add                 # every replay/import/*.txt (level strings), renamed to .saved
    python scripts/gd_save.py add --wait          # keep waiting: add pending levels whenever the game is closed
    python scripts/gd_save.py list                # the levels in the save
    python scripts/gd_save.py restore <backup dir>
"""
from __future__ import annotations

import argparse
import base64
import gzip
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMPORT = ROOT / "replay" / "import"
BACKUPS = ROOT / "replay" / "save_backup"


def save_dir() -> Path:
    local = subprocess.run(["cmd.exe", "/c", "echo %LOCALAPPDATA%"], capture_output=True, text=True).stdout.strip()
    return Path(subprocess.run(["wslpath", local], capture_output=True, text=True).stdout.strip()) / "GeometryDash"


def game_running() -> bool:
    out = subprocess.run(["tasklist.exe", "/FI", "IMAGENAME eq GeometryDash.exe"], capture_output=True)
    return b"GeometryDash.exe" in out.stdout          # bytes: tasklist prints in the console code page


def decode(raw: bytes) -> str:
    x = bytes(b ^ 11 for b in raw).rstrip(b"\x00")
    return zlib.decompress(base64.urlsafe_b64decode(x + b"=" * (-len(x) % 4)), 15 | 32).decode()


def encode(xml: str) -> bytes:
    return bytes(b ^ 11 for b in base64.urlsafe_b64encode(gzip.compress(xml.encode(), compresslevel=6)))


def entries(d: ET.Element) -> list[tuple[ET.Element, ET.Element]]:
    kids = list(d)
    return [(kids[i], kids[i + 1]) for i in range(0, len(kids) - 1, 2)]


def level_array(root: ET.Element) -> ET.Element:
    top = root.find("dict")
    for k, v in entries(top):
        if k.text == "LLM_01":
            return v
    raise ValueError("no LLM_01 (local levels) in the save")


def binary_version(root: ET.Element) -> str:
    """LLM_02: the game version that wrote the save (47 for 2.2081, 51 for the October 2026 build)."""
    return next((v.text for k, v in entries(root.find("dict")) if k.text == "LLM_02"), "47")


def level_dict(name: str, level_string: str, version: str) -> ET.Element:
    d = ET.Element("d")

    def put(key: str, tag: str, text: str | None = None) -> None:
        ET.SubElement(d, "k").text = key
        el = ET.SubElement(d, tag)
        if text is not None:
            el.text = text

    put("kCEK", "i", "4")                   # GJGameLevel
    put("k2", "s", name)
    put("k4", "s", level_string)
    put("k5", "s", "gdgen")
    put("k13", "t")
    put("k21", "i", "2")                    # editor (local) level
    put("k16", "i", "1")
    put("k50", "i", version)
    return d


def add_levels(xml: str, levels: list[tuple[str, str]]) -> str:
    root = ET.fromstring(xml)
    arr = level_array(root)
    old = [v for k, v in entries(arr) if k.text and k.text.startswith("k_")]
    for el in list(arr):
        arr.remove(el)
    ET.SubElement(arr, "k").text = "_isArr"
    ET.SubElement(arr, "t")
    version = binary_version(root)
    for i, d in enumerate([level_dict(n, s, version) for n, s in levels] + old):
        ET.SubElement(arr, "k").text = f"k_{i}"
        arr.append(d)
    return '<?xml version="1.0"?>' + ET.tostring(root, encoding="unicode")


def backup(sd: Path) -> Path:
    dest = BACKUPS / time.strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("CCLocalLevels.dat", "CCLocalLevels2.dat"):
        if (sd / f).exists():
            shutil.copy2(sd / f, dest / f)
    return dest


def add_pending(sd: Path) -> int:
    files = sorted(IMPORT.glob("*.txt"))
    if not files:
        return 0
    if game_running():
        raise SystemExit("Geometry Dash is running: close it first (it rewrites the save when it closes)")
    levels = [(f.stem, f.read_text().strip()) for f in files]
    path = sd / "CCLocalLevels.dat"
    xml = decode(path.read_bytes())
    new = add_levels(xml, levels)
    decode(encode(new))                     # the written file must read back
    dest = backup(sd)
    path.write_bytes(encode(new))
    for f in files:
        f.rename(f.with_suffix(".saved"))
    print(time.strftime("%H:%M:%S"), f"added {len(levels)} levels ({', '.join(n for n, _ in levels)}); backup {dest}",
          flush=True)
    return len(levels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["add", "list", "restore"])
    ap.add_argument("backup", nargs="?")
    ap.add_argument("--wait", action="store_true", help="add: keep going, whenever the game is closed")
    args = ap.parse_args()
    sd = save_dir()
    if args.action == "list":
        arr = level_array(ET.fromstring(decode((sd / "CCLocalLevels.dat").read_bytes())))
        for k, d in entries(arr):
            if k.text.startswith("k_"):
                name = next((v.text for kk, v in entries(d) if kk.text == "k2"), "?")
                size = len(next((v.text or "" for kk, v in entries(d) if kk.text == "k4"), ""))
                print(f"{k.text:6s} {name}  ({size // 1024} KB level string)")
        return 0
    if args.action == "restore":
        if game_running():
            raise SystemExit("close Geometry Dash first")
        for f in Path(args.backup).glob("CCLocalLevels*.dat"):
            shutil.copy2(f, sd / f.name)
            print("restored", sd / f.name)
        return 0
    if not args.wait:
        add_pending(sd)
        return 0
    print("waiting for levels in", IMPORT, "and for Geometry Dash to be closed", flush=True)
    while True:                             # the process is gone only after the game wrote its own save
        if any(IMPORT.glob("*.txt")) and not game_running():
            add_pending(sd)
        time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
