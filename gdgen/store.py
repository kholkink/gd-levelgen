"""SQLite store for crawled levels. One file, resumable, safe to read while the crawler writes."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

LENGTH_PLATFORMER = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS levels (
    id            INTEGER PRIMARY KEY,
    name          TEXT,
    player_id     INTEGER,
    game_version  INTEGER,
    length        INTEGER,   -- 0 tiny .. 4 XL, 5 platformer
    stars         INTEGER,
    difficulty    INTEGER,   -- numerator, /10 -> 0 NA,1 easy,..,5 insane
    demon         INTEGER,
    demon_diff    INTEGER,
    feature_score INTEGER,
    epic          INTEGER,   -- 0 none, 1 epic, 2 legendary, 3 mythic
    downloads     INTEGER,
    likes         INTEGER,
    objects       INTEGER,
    two_player    INTEGER,
    song_id       INTEGER,   -- custom song id, 0 if official
    official_song INTEGER,
    list_meta     TEXT,      -- JSON: fields from the search listing
    dl_meta       TEXT,      -- JSON: fields from the download, minus the level string
    data          BLOB,      -- compressed (gzip/zlib) raw level text
    data_len      INTEGER,   -- decompressed length in bytes
    status        TEXT NOT NULL,  -- listed | downloaded | skipped | missing | error
    attempts      INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    listed_at     REAL,
    fetched_at    REAL
);
CREATE INDEX IF NOT EXISTS levels_status ON levels(status);
CREATE TABLE IF NOT EXISTS creators (player_id INTEGER PRIMARY KEY, name TEXT, account_id INTEGER);
CREATE TABLE IF NOT EXISTS songs (id INTEGER PRIMARY KEY, meta TEXT);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""


def _int(meta: dict[str, str], key: str) -> int | None:
    v = meta.get(key, "")
    try:
        return int(v)
    except ValueError:
        return None


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A long busy timeout lets writes wait out a reader such as a `VACUUM INTO` snapshot (minutes on a big file).
        self.db = sqlite3.connect(str(self.path), timeout=3600)
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def get_state(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_state(self, key: str, value) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO state(key, value) VALUES (?, ?)", (key, str(value)))

    def add_listing(self, levels, creators, songs, skip_platformer: bool) -> list[int]:
        """Upserts search results; returns ids that still need downloading."""
        now = time.time()
        with self.db:
            for pid, c in creators.items():
                self.db.execute(
                    "INSERT OR REPLACE INTO creators(player_id, name, account_id) VALUES (?, ?, ?)",
                    (pid, c["name"], _int(c, "account_id")),
                )
            for s in songs:
                if s.get("1", "").isdigit():
                    self.db.execute("INSERT OR REPLACE INTO songs(id, meta) VALUES (?, ?)", (int(s["1"]), json.dumps(s)))
            for m in levels:
                lid = int(m["1"])
                status = "skipped" if skip_platformer and _int(m, "15") == LENGTH_PLATFORMER else "listed"
                self.db.execute(
                    """INSERT INTO levels(id, name, player_id, game_version, length, stars, difficulty, demon,
                           demon_diff, feature_score, epic, downloads, likes, objects, two_player, song_id,
                           official_song, list_meta, status, listed_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET
                           list_meta = excluded.list_meta, feature_score = excluded.feature_score,
                           epic = excluded.epic, downloads = excluded.downloads, likes = excluded.likes,
                           status = CASE WHEN levels.status = 'skipped' AND excluded.status = 'listed'
                                         THEN 'listed' ELSE levels.status END""",
                    (lid, m.get("2"), _int(m, "6"), _int(m, "13"), _int(m, "15"), _int(m, "18"), _int(m, "9"),
                     _int(m, "17"), _int(m, "43"), _int(m, "19"), _int(m, "42"), _int(m, "10"), _int(m, "14"),
                     _int(m, "45"), _int(m, "31"), _int(m, "35"), _int(m, "12"), json.dumps(m), status, now),
                )
            ids = [int(m["1"]) for m in levels]
            if not ids:
                return []
            q = f"SELECT id FROM levels WHERE status = 'listed' AND id IN ({','.join('?' * len(ids))})"
            pending = {r[0] for r in self.db.execute(q, ids)}
        return [i for i in ids if i in pending]

    def save_download(self, level_id: int, meta: dict[str, str], blob: bytes, data_len: int) -> None:
        dl_meta = {k: v for k, v in meta.items() if k != "4"}
        with self.db:
            self.db.execute(
                """UPDATE levels SET data = ?, data_len = ?, dl_meta = ?, status = 'downloaded',
                       objects = COALESCE(?, objects), attempts = attempts + 1, error = NULL, fetched_at = ?
                   WHERE id = ?""",
                (blob, data_len, json.dumps(dl_meta), _int(meta, "45"), time.time(), level_id),
            )

    def mark(self, level_id: int, status: str, error: str | None = None) -> None:
        with self.db:
            self.db.execute(
                "UPDATE levels SET status = ?, error = ?, attempts = attempts + 1, fetched_at = ? WHERE id = ?",
                (status, error, time.time(), level_id),
            )

    def retry_candidates(self, max_attempts: int) -> list[int]:
        rows = self.db.execute(
            """SELECT id FROM levels
               WHERE status = 'listed' OR (status IN ('error', 'missing') AND attempts < ?)
               ORDER BY feature_score DESC""",
            (max_attempts,),
        )
        return [r[0] for r in rows]

    def counts(self) -> dict[str, int]:
        return dict(self.db.execute("SELECT status, COUNT(*) FROM levels GROUP BY status").fetchall())

    def data_bytes(self) -> int:
        return self.db.execute("SELECT COALESCE(SUM(LENGTH(data)), 0) FROM levels").fetchone()[0]
