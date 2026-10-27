"""Tokenizes downloaded levels into uint16 shards for training (incremental: already indexed levels are skipped).

Per level two streams are written; together they hold every object of the level:
    layout  `encode_level` of the gameplay objects: <BOS> header <OBJECTS> objects <EOS>
            (gameplay = not decoration, triggers, coins, start positions or no-touch objects)
    deco    everything else, objects only: <OBJECTS> objects <EOS>
Both streams carry SEG tokens, so the training loader can cut aligned segment ranges from them. It inserts the
stream token and the M:* metadata tokens after <BOS>.

Layout of data/tokens/v1/:
    layout_NNN.bin, deco_NNN.bin    concatenated uint16 token ids
    index.jsonl                     one line per level: {"id", "shard", "layout": [offset, length],
                                    "deco": [offset, length], "header_len", "objects", "gameplay",
                                    "stars", "demon", "epic", "length", "game_version", "song_id", "two_player"}
    vocab.json                      the vocabulary the ids refer to

Usage: python scripts/pretokenize.py [--workers 8] [--limit N] [--db levels.sqlite] [--part i/n]
--part i/n tokenizes only levels with id % n == i (parallel Kaggle kernels, each with its own --out).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gdgen.levelstring import decompress, parse_level  # noqa: E402
from gdgen.objects import is_layout  # noqa: E402
from gdgen.music import level_music, load_features  # noqa: E402
from gdgen.tokens import Tokenizer, Vocab  # noqa: E402

DB = ROOT / "data" / "levels.sqlite"
DB_URI = f"file:{DB}?mode=ro"
OUT = ROOT / "data" / "tokens" / "v1"
SHARD_TOKENS = 500_000_000            # ~1 GB per deco shard
DEMON_TIER = {3: 1, 4: 2, 0: 3, 5: 4, 6: 5}  # server demon_diff -> easy..extreme demon (0 = not a demon)

TOK: Tokenizer | None = None
FEATURES: dict[int, str] = {}                 # song id -> song features .npz (scripts/song_features.py)


def encode(row) -> dict:
    level_id, blob, stars, demon, demon_diff, epic, length, game_version, song_id, two_player = row
    header, objects = parse_level(decompress(blob))
    music = None
    if song_id in FEATURES:
        try:
            music = level_music(header, objects, load_features(FEATURES[song_id]))
        except (OSError, ValueError, KeyError):
            music = None
    flags = [is_layout(o) for o in objects]
    layout = TOK.encode_level(header, [o for o, g in zip(objects, flags) if g], music)
    deco = ["<OBJECTS>", *TOK.encode_objects([o for o, g in zip(objects, flags) if not g], music), "<EOS>"]
    gameplay = sum(flags)
    return {
        "id": level_id,
        "layout": np.asarray(TOK.vocab.ids(layout), dtype=np.uint16),
        "deco": np.asarray(TOK.vocab.ids(deco), dtype=np.uint16),
        "header_len": layout.index("<OBJECTS>") + 1,
        "objects": len(objects), "gameplay": gameplay,
        "stars": stars or 0, "demon": DEMON_TIER.get(demon_diff, 3) if demon else 0, "epic": epic or 0,
        "length": length or 0, "game_version": game_version or 0, "song_id": song_id or 0,
        "two_player": two_player or 0, "unk": layout.count("<UNK>") + deco.count("<UNK>"), "music": bool(music),
    }


def init(vocab_path: str, features: dict[int, str] | None = None) -> None:
    global TOK
    TOK = Tokenizer(Vocab.load(vocab_path))
    FEATURES.update(features or {})


def feature_files(root: str | None) -> dict[int, str]:
    """Every <song id>.npz under root (the song features kernels' outputs)."""
    if not root:
        return {}
    return {int(p.stem): str(p) for p in Path(root).rglob("*.npz") if p.stem.isdigit()}


def db_uri(path: str) -> str:
    """Read-only URI; a snapshot (no crawler writing to it, e.g. on Kaggle's read-only input) is opened immutable."""
    return f"file:{path}?mode=ro" + ("&immutable=1" if Path(path).resolve() != DB.resolve() else "")


def rows(ids: list[int], uri: str):
    db = sqlite3.connect(uri, uri=True, timeout=60)
    for level_id in ids:
        yield db.execute(
            "SELECT id, data, stars, demon, demon_diff, epic, length, game_version, song_id, two_player "
            "FROM levels WHERE id = ?", (level_id,)).fetchone()
    db.close()


class ShardWriter:
    def __init__(self, out: Path, first_shard: int):
        self.out = out
        self.shard = first_shard - 1
        self.files: dict[str, object] = {}
        self.pos = {"layout": 0, "deco": 0}
        self.rotate()

    def rotate(self) -> None:
        for f in self.files.values():
            f.close()
        self.shard += 1
        self.files = {s: open(self.out / f"{s}_{self.shard:03d}.bin", "ab") for s in ("layout", "deco")}
        self.pos = {"layout": 0, "deco": 0}

    def write(self, stream: str, arr: np.ndarray) -> list[int]:
        off = self.pos[stream]
        self.files[stream].write(arr.tobytes())
        self.pos[stream] += len(arr)
        return [off, len(arr)]

    def close(self) -> None:
        for f in self.files.values():
            f.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vocab", default=str(ROOT / "gdgen" / "data" / "vocab_v1.json"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--part", default="0/1", help="i/n: only levels with id %% n == i")
    ap.add_argument("--features", help="directory with song feature .npz files: adds music tokens after SEG")
    args = ap.parse_args()
    part, nparts = map(int, args.part.split("/"))
    uri = db_uri(args.db)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.jsonl"
    done, shards = set(), [-1]
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            rec = json.loads(line)
            done.add(rec["id"])
            shards.append(rec["shard"])
        if (out / "vocab.json").read_text() != Path(args.vocab).read_text():
            sys.exit("vocab changed since these shards were written; use a new --out")
    (out / "vocab.json").write_text(Path(args.vocab).read_text())

    db = sqlite3.connect(uri, uri=True, timeout=60)
    todo = [r[0] for r in db.execute("SELECT id FROM levels WHERE status = 'downloaded' ORDER BY id")
            if r[0] not in done and r[0] % nparts == part]
    db.close()
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} levels already tokenized, {len(todo)} to go", flush=True)

    writer = ShardWriter(out, max(shards) + 1)     # never append to a shard a crash may have left half-written
    t0, n_tok = time.time(), 0
    with open(index_path, "a") as index, \
            Pool(args.workers, initializer=init, initargs=(args.vocab, feature_files(args.features)),
                 maxtasksperchild=50) as pool:
        for n, r in enumerate(pool.imap_unordered(encode, rows(todo, uri), chunksize=2), 1):
            if writer.pos["deco"] + len(r["deco"]) > SHARD_TOKENS and writer.pos["deco"] > 0:
                writer.rotate()
            layout, deco = r.pop("layout"), r.pop("deco")
            rec = {**r, "shard": writer.shard, "layout": writer.write("layout", layout),
                   "deco": writer.write("deco", deco)}
            for f in writer.files.values():
                f.flush()
            index.write(json.dumps(rec) + "\n")
            index.flush()
            n_tok += len(layout) + len(deco)
            if n % 200 == 0 or n == len(todo):
                dt = time.time() - t0
                print(f"{n}/{len(todo)} levels, {n_tok / 1e6:.0f}M tokens, {n / dt:.1f} levels/s, "
                      f"ETA {(len(todo) - n) / max(n / dt, 1e-9) / 60:.0f} min", flush=True)
    writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
