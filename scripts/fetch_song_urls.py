"""Collects download URLs for the custom songs of the downloaded levels (resumable, polite).

Music-library songs (id >= 10,000,000) live at a fixed CDN address; songs whose listing metadata already holds a
URL use it; the rest are asked from the GD server (getGJSongInfo), one request every --delay seconds, backing off
on rate limits like the crawler. The audio itself is fetched later by the Kaggle songs kernel.

Writes data/songs/song_urls.jsonl, one line per song:
    {"id", "url" (null when the song is gone), "source": library|listing|info|missing, "name", "artist", "mb"}

Usage: python scripts/fetch_song_urls.py [--delay 4]
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gdgen.server import GDClient, RateLimited, ServerError, parse_kv  # noqa: E402

DB = ROOT / "data" / "levels.sqlite"
OUT = ROOT / "data" / "songs" / "song_urls.jsonl"
LIBRARY_MIN_ID = 10_000_000
LIBRARY_URL = "https://geometrydashfiles.b-cdn.net/music/{id}.ogg"
log = logging.getLogger("songs")


def song_record(sid: int, meta: dict, source: str) -> dict:
    url = urllib.parse.unquote(meta.get("10", "") or "")
    try:
        mb = float(meta.get("5", "") or 0)
    except ValueError:
        mb = 0.0
    return {"id": sid, "url": url if url.startswith("http") else None, "source": source,
            "name": meta.get("2", ""), "artist": meta.get("4", ""), "mb": mb}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=4.0)
    ap.add_argument("--max-backoff", type=float, default=1800)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=120)
    ids = [r[0] for r in db.execute(
        "SELECT song_id FROM levels WHERE status = 'downloaded' AND song_id > 0 GROUP BY song_id ORDER BY COUNT(*) DESC")]
    metas = {r[0]: json.loads(r[1]) for r in db.execute("SELECT id, meta FROM songs")}
    db.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(line)["id"] for line in OUT.read_text().splitlines() if line}
    todo = [i for i in ids if i not in done]
    log.info("%d songs, %d already resolved, %d to go", len(ids), len(done), len(todo))

    client = GDClient(delay=args.delay)
    backoff = 0.0
    with open(OUT, "a") as out:
        for n, sid in enumerate(todo, 1):
            meta = metas.get(sid, {})
            if sid >= LIBRARY_MIN_ID:
                rec = song_record(sid, meta, "library")
                rec["url"] = LIBRARY_URL.format(id=sid)
            elif song_record(sid, meta, "listing")["url"]:
                rec = song_record(sid, meta, "listing")
            else:
                while True:
                    try:
                        text = client.post("getGJSongInfo.php", songID=str(sid))
                        backoff = 0.0
                        break
                    except RateLimited as e:
                        backoff = min(max(backoff * 2, 60), args.max_backoff)
                        log.warning("rate limited (%s); pausing %ds", e, backoff)
                    except ServerError as e:
                        backoff = min(max(backoff * 2, 15), args.max_backoff)
                        log.warning("server error (%s); retry in %ds", e, backoff)
                    time.sleep(backoff)
                if text.strip() in ("-1", "-2", ""):
                    rec = song_record(sid, meta, "missing")
                else:
                    rec = song_record(sid, parse_kv(text, "~|~"), "info")
                    if rec["url"] is None:
                        rec["source"] = "missing"
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if n % 100 == 0:
                log.info("%d/%d songs resolved (%d requests)", n, len(todo), client.n_requests)
    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
