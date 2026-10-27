"""Renders levels to PNG previews with the game's sprites (gdgen/render.py; needs the local GD install).

    python scripts/render_level.py --id 108777464 --out preview.png          # a downloaded level
    python scripts/render_level.py --file gen/try1/level.txt --out try1.png   # raw level text (generate.py output)
    python scripts/render_level.py --file gen/try1/level.txt --out try1.png --rows 6 --scale 0.4
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gdgen.levelstring import decompress, level_text, parse_level  # noqa: E402
from gdgen.render import Renderer  # noqa: E402

DB = ROOT / "data" / "levels.sqlite"


def load(args) -> tuple[str, dict, list[dict]]:
    if args.id:
        db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
        row = db.execute("SELECT name, data FROM levels WHERE id = ? AND status = 'downloaded'", (args.id,)).fetchone()
        if row is None:
            raise SystemExit(f"level {args.id} is not downloaded")
        return row[0], *parse_level(decompress(row[1]))
    return Path(args.file).stem, *parse_level(level_text(Path(args.file).read_text()))


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--id", type=int)
    src.add_argument("--file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=8, help="at most this many rows of --row-units each")
    ap.add_argument("--row-units", type=float, default=2400.0)
    ap.add_argument("--row-height", type=float, default=720.0)
    ap.add_argument("--scale", type=float, default=0.5, help="output pixels per game unit")
    args = ap.parse_args()
    name, header, objects = load(args)
    t0 = time.time()
    renderer = Renderer()
    img = renderer.render(header, objects, scale=args.scale, row_units=args.row_units, row_height=args.row_height,
                          max_rows=args.rows)
    img.save(args.out)
    missing = sum(renderer.missing.values())
    print(f"{name}: {len(objects)} objects -> {args.out} {img.size[0]}x{img.size[1]} in {time.time() - t0:.1f}s"
          + (f" ({missing} objects without a sprite)" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
