"""Statistics over downloaded levels that drive the tokenizer design.

For a random sample of levels it reports, per object category: how often each object key is set, how many
distinct values it takes and the most common ones; how positions sit on the grid; and level sizes.

Usage: python scripts/level_stats.py [--sample 400] [--workers 6] [--out data/stats/level_stats.json]
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gdgen.levelstring import decompress, parse_level  # noqa: E402
from gdgen.objects import category_of  # noqa: E402

DB = ROOT / "data" / "levels.sqlite"
MAX_DISTINCT = 4000          # per key and category; beyond that only the count of objects is kept


def grid_class(v: float) -> str:
    """How finely a coordinate is placed: on the 15-unit half grid, the 2-unit nudge grid, 0.5, 0.25, or free."""
    for name, step in (("15", 15.0), ("7.5", 7.5), ("2", 2.0), ("1", 1.0), ("0.5", 0.5), ("0.25", 0.25), ("0.1", 0.1)):
        if abs(v / step - round(v / step)) < 1e-6:
            return name
    return "free"


def level_stats(level_id: int) -> dict:
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    blob, = db.execute("SELECT data FROM levels WHERE id = ?", (level_id,)).fetchone()
    db.close()
    header, objects = parse_level(decompress(blob))
    keys: dict[str, Counter] = defaultdict(Counter)          # "cat|key" -> value counter
    key_objs: Counter = Counter()                           # "cat|key" -> objects having it
    cat_count: Counter = Counter()
    ids: Counter = Counter()
    grid: dict[str, Counter] = defaultdict(Counter)
    for o in objects:
        try:
            oid = int(o.get("1", "0"))
        except ValueError:
            continue
        cat = category_of(oid)
        cat_count[cat] += 1
        ids[oid] += 1
        group = "gameplay" if cat not in ("deco", "trigger", "unknown", "collectible") else cat
        for axis, k in (("x", "2"), ("y", "3")):
            try:
                grid[f"{group}|{axis}"][grid_class(float(o.get(k, "0")))] += 1
            except ValueError:
                pass
        for k, v in o.items():
            if k in ("1", "2", "3"):
                continue
            ck = f"{group}|{k}"
            key_objs[ck] += 1
            if len(keys[ck]) < MAX_DISTINCT or v in keys[ck]:
                keys[ck][v] += 1
    xs = [float(o.get("2", 0) or 0) for o in objects if category_of(int(o.get("1", "0") or 0)) not in ("deco", "trigger")]
    return {
        "id": level_id, "objects": len(objects), "cats": dict(cat_count), "ids": dict(ids),
        "key_objs": dict(key_objs), "keys": {k: dict(c) for k, c in keys.items()},
        "grid": {k: dict(c) for k, c in grid.items()}, "header_keys": sorted(header),
        "length_x": max(xs) if xs else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "stats" / "level_stats.json"))
    args = ap.parse_args()

    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    ids = [r[0] for r in db.execute("SELECT id FROM levels WHERE status = 'downloaded'")]
    db.close()
    random.Random(args.seed).shuffle(ids)
    sample = ids[: args.sample]

    key_objs: Counter = Counter()
    keys: dict[str, Counter] = defaultdict(Counter)
    cats: Counter = Counter()
    id_levels: Counter = Counter()      # in how many levels an object id appears
    id_count: Counter = Counter()
    grid: dict[str, Counter] = defaultdict(Counter)
    header_keys: Counter = Counter()
    sizes = []
    with Pool(args.workers, maxtasksperchild=20) as pool:
        for n, s in enumerate(pool.imap_unordered(level_stats, sample), 1):
            key_objs.update(s["key_objs"])
            for k, c in s["keys"].items():
                keys[k].update(c)
            cats.update(s["cats"])
            id_levels.update(s["ids"].keys())
            id_count.update(s["ids"])
            for k, c in s["grid"].items():
                grid[k].update(c)
            header_keys.update(s["header_keys"])
            sizes.append({"id": s["id"], "objects": s["objects"], "length_x": s["length_x"], **s["cats"]})
            if n % 50 == 0:
                print(f"{n}/{len(sample)}", file=sys.stderr)

    out = {
        "levels": len(sample), "cats": dict(cats), "sizes": sizes,
        "grid": {k: dict(c) for k, c in grid.items()},
        "header_keys": dict(header_keys.most_common()),
        "id_count": {str(k): v for k, v in id_count.most_common()},
        "id_levels": {str(k): v for k, v in id_levels.most_common()},
        "keys": {
            k: {"objects": key_objs[k], "distinct": len(keys[k]), "top": keys[k].most_common(40)}
            for k in sorted(key_objs, key=lambda k: -key_objs[k])
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
