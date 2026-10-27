"""Fits the token vocabulary (gdgen/data/vocab_v1.json) on a sample of downloaded levels.

Pass 1 decides each object key's kind (flag, list, hsv, ...) and which non-integer values get their own V token;
pass 2 (which needs the kinds) finds the frequent flag sets and key=value pairs that get single tokens.

Usage: python scripts/fit_vocab.py [--sample 1500] [--workers 6] [--max-float-values 8000]
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gdgen.levelstring import decompress, parse_level  # noqa: E402
from gdgen.tokens import VOCAB_FILE, KindCounter, PairCounter, build_vocab  # noqa: E402

DB = ROOT / "data" / "levels.sqlite"
KINDS: dict[str, str] = {}


def load_level(level_id: int):
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    blob, = db.execute("SELECT data FROM levels WHERE id = ?", (level_id,)).fetchone()
    db.close()
    return parse_level(decompress(blob))


def count_kinds(level_id: int) -> KindCounter:
    header, objects = load_level(level_id)
    c = KindCounter()
    c.add_header(header)
    for o in objects:
        c.add_object(o)
    return c


def count_pairs(level_id: int) -> PairCounter:
    _, objects = load_level(level_id)
    c = PairCounter(KINDS)
    for o in objects:
        c.add_object(o)
    return c


def set_kinds(kinds: dict[str, str]) -> None:
    KINDS.update(kinds)


def sample_ids(n: int, seed: int) -> list[int]:
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    ids = sorted(r[0] for r in db.execute("SELECT id FROM levels WHERE status = 'downloaded'"))
    db.close()
    random.Random(seed).shuffle(ids)
    return ids[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-float-values", type=int, default=8000)
    ap.add_argument("--max-pairs", type=int, default=22000)
    ap.add_argument("--max-flagsets", type=int, default=1024)
    ap.add_argument("--out", default=str(VOCAB_FILE))
    args = ap.parse_args()

    total = KindCounter()
    ids = sample_ids(args.sample, args.seed)
    with Pool(args.workers, maxtasksperchild=20) as pool:
        for n, c in enumerate(pool.imap_unordered(count_kinds, ids), 1):
            total.merge(c)
            if n % 100 == 0:
                print(f"pass 1: {n}/{len(ids)} levels", file=sys.stderr)
    kinds = total.kinds()
    pairs = PairCounter(kinds)
    with Pool(args.workers, initializer=set_kinds, initargs=(kinds,), maxtasksperchild=20) as pool:
        for n, c in enumerate(pool.imap_unordered(count_pairs, ids), 1):
            pairs.merge(c)
            if n % 100 == 0:
                print(f"pass 2: {n}/{len(ids)} levels", file=sys.stderr)
    vocab = build_vocab(total, pairs, max_float_values=args.max_float_values, max_pairs=args.max_pairs,
                        max_flagsets=args.max_flagsets)
    vocab.save(args.out)
    by_kind: dict[str, list[str]] = {}
    for k, kind in vocab.key_kinds.items():
        by_kind.setdefault(kind, []).append(k)
    for kind, ks in sorted(by_kind.items()):
        print(f"{kind:7s} {len(ks):4d} keys: {' '.join(sorted(ks, key=int)[:40])}{' ...' if len(ks) > 40 else ''}")
    covered = sum(n for _, n in total.values.most_common(args.max_float_values))
    print(f"non-integer values: {len(total.values)} distinct; top {args.max_float_values} cover "
          f"{covered / max(1, sum(total.values.values())):.4f} of occurrences")
    for name, counter, top in (("pairs", pairs.pairs, args.max_pairs), ("flag sets", pairs.flagsets, args.max_flagsets)):
        cov = sum(n for _, n in counter.most_common(top)) / max(1, sum(counter.values()))
        print(f"{name}: {len(counter)} distinct; top {top} cover {cov:.4f}")
    print(f"vocab: {len(vocab)} tokens -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
