"""Tokenizer report on a sample of downloaded levels: round-trip check and where the tokens go.

For every sampled level it encodes, decodes and compares with the canonical form (the round-trip contract of
gdgen/tokens.py), then breaks the token count down by object group (gameplay / deco / trigger) and by token
role (id, position, groups, flags, values, digit fallback, ...).

Usage: python scripts/token_report.py [--sample 200] [--workers 6]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fit_vocab import DB, sample_ids  # noqa: E402

from gdgen.levelstring import decompress, parse_level  # noqa: E402
from gdgen.objects import category_of  # noqa: E402
from gdgen.tokens import Tokenizer, Vocab, canonical_header, canonical_object  # noqa: E402

TOK: Tokenizer | None = None


def role(t: str) -> str:
    if t.startswith(("X:", "Y:", "YF:", "SEG:")):
        return "position"
    if t.startswith(("O:", "O?")):
        return "id"
    if t.startswith("G:"):
        return "groups"
    if t.startswith(("F:", "F0:")):
        return "flags"
    if t.startswith("K:"):
        return "keys"
    if t.startswith("V:"):
        return "values"
    if t.startswith(("D:", "DEND")):
        return "digits"
    if t.startswith(("HH:", "HS:", "HV:")):
        return "hsv"
    if t.startswith("B:") or t in ("TRAW", "LEND"):
        return "bytes/lists"
    return "other"


def group_of(oid: int) -> str:
    cat = category_of(oid)
    return cat if cat in ("deco", "trigger") else "gameplay"


def report_level(level_id: int) -> dict:
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    blob, = db.execute("SELECT data FROM levels WHERE id = ?", (level_id,)).fetchone()
    db.close()
    header, objects = parse_level(decompress(blob))
    tokens = TOK.encode_level(header, objects)
    dh, dobjs = TOK.decode_level(tokens)

    canon = [c for c in (canonical_object(o, TOK.kinds) for o in objects) if c is not None]
    a = sorted(tuple(sorted(o.items())) for o in canon)
    b = sorted(tuple(sorted(o.items())) for o in dobjs)
    mism = abs(len(a) - len(b)) + sum(1 for x, y in zip(a, b) if x != y)
    example = None
    if mism:
        sa = set(a)
        diff = [o for o in dobjs if tuple(sorted(o.items())) not in sa][:1]
        if diff:
            pos = lambda o: (float(o["2"]), float(o["3"]))  # noqa: E731
            d = pos(diff[0])
            near = min(canon, key=lambda o: abs(pos(o)[0] - d[0]) + abs(pos(o)[1] - d[1]))
            example = {"decoded": diff[0], "canonical": near}
    header_ok = dh == canonical_header(header)

    by_group: Counter = Counter()
    by_role: Counter = Counter()
    objs_by_group: Counter = Counter()
    cur = "header"
    for t in tokens:
        if t.startswith("O:"):
            cur = group_of(int(t[2:]))
            objs_by_group[cur] += 1
        elif t == "O?":
            cur = "gameplay"
        elif t.startswith("SEG:"):
            cur = "seg"
        by_group[cur] += 1
        by_role[f"{cur}|{role(t)}"] += 1
    return {"id": level_id, "tokens": len(tokens), "objects": len(dobjs), "mismatch": mism, "header_ok": header_ok,
            "example": example, "by_group": dict(by_group), "by_role": dict(by_role), "objs": dict(objs_by_group),
            "unk": tokens.count("<UNK>")}


def init(vocab_path: str) -> None:
    global TOK
    TOK = Tokenizer(Vocab.load(vocab_path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--vocab", default=str(ROOT / "gdgen" / "data" / "vocab_v1.json"))
    args = ap.parse_args()

    ids = sample_ids(args.sample, args.seed)
    with Pool(args.workers, initializer=init, initargs=(args.vocab,), maxtasksperchild=20) as pool:
        rows = list(pool.imap_unordered(report_level, ids))

    bad = [r for r in rows if r["mismatch"] or not r["header_ok"] or r["unk"]]
    print(f"levels: {len(rows)}, round-trip failures: {len(bad)}")
    for r in bad[:5]:
        print(f"  level {r['id']}: {r['mismatch']} objects differ, header_ok={r['header_ok']}, unk={r['unk']}")
        if r["example"]:
            print(f"    decoded:   {r['example']['decoded']}")
            print(f"    canonical: {r['example']['canonical']}")

    toks = sorted(r["tokens"] for r in rows)
    print(f"tokens/level p10/p50/p90: {toks[len(toks) // 10]:,} / {toks[len(toks) // 2]:,} / "
          f"{toks[9 * len(toks) // 10]:,}   total {sum(toks):,}")
    by_group: Counter = Counter()
    by_role: Counter = Counter()
    objs: Counter = Counter()
    for r in rows:
        by_group.update(r["by_group"])
        by_role.update(r["by_role"])
        objs.update(r["objs"])
    total = sum(by_group.values())
    for g in ("gameplay", "deco", "trigger"):
        roles = sorted(((k.split("|")[1], v) for k, v in by_role.items() if k.startswith(g + "|")), key=lambda kv: -kv[1])
        print(f"{g:9s} {by_group[g] / total:6.1%} of tokens, {by_group[g] / max(1, objs[g]):5.2f} tokens/object; "
              + ", ".join(f"{k} {v / max(1, objs[g]):.2f}" for k, v in roles))
    print(f"header+seg {(by_group['header'] + by_group['seg']) / total:6.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
