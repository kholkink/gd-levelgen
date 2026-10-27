"""Resumable crawler: walks a server listing (featured by default) page by page and downloads every level.

Progress lives in the SQLite store, so the process can be killed and restarted at any time.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import zlib
from pathlib import Path

from .levelstring import to_storage_bytes
from .server import SEARCH_FEATURED, GDClient, RateLimited, ServerError
from .store import Store

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PID_FILE = DATA / "crawl.pid"
PROGRESS_FILE = DATA / "progress.json"

log = logging.getLogger("crawl")


class Stopped(Exception):
    pass


class Crawler:
    def __init__(self, store: Store, client: GDClient, max_backoff: int = 1800, missing_streak_pause: int = 600):
        self.store = store
        self.client = client
        self.max_backoff = max_backoff
        self.missing_streak_pause = missing_streak_pause
        self.stop = False
        self.backoff = 0
        self.missing_streak = 0
        self.downloaded_now = 0

    def request_stop(self, *_):
        log.info("stop requested")
        self.stop = True

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.stop:
                raise Stopped
            time.sleep(min(1.0, end - time.monotonic()))

    def call(self, fn, *args, max_errors: int | None = None):
        """Retries rate limits forever (with growing pauses) and server errors up to max_errors times."""
        errors = 0
        while True:
            if self.stop:
                raise Stopped
            try:
                result = fn(*args)
                self.backoff = 0
                return result
            except RateLimited as e:
                self.backoff = min(max(self.backoff * 2, 60), self.max_backoff)
                log.warning("rate limited (%s); pausing %ds", e, self.backoff)
            except ServerError as e:
                errors += 1
                if max_errors is not None and errors >= max_errors:
                    raise
                self.backoff = min(max(self.backoff * 2, 15), self.max_backoff)
                log.warning("server error (%s); retry in %ds", e, self.backoff)
            self._sleep(self.backoff)

    def download_one(self, level_id: int) -> None:
        try:
            meta = self.call(self.client.download_level, level_id, max_errors=3)
        except ServerError as e:
            log.error("level %d failed: %s", level_id, e)
            self.store.mark(level_id, "error", str(e)[:500])
            return
        if meta is None:
            self.store.mark(level_id, "missing")
            self.missing_streak += 1
            if self.missing_streak >= 5:
                # Several "-1" in a row for featured levels smells like a soft block, not deletions.
                log.warning("%d missing levels in a row; pausing %ds", self.missing_streak, self.missing_streak_pause)
                self._sleep(self.missing_streak_pause)
                self.missing_streak = 0
            return
        self.missing_streak = 0
        try:
            blob = to_storage_bytes(meta["4"])
            data_len = len(zlib.decompress(blob, 15 | 32))
        except (ValueError, zlib.error) as e:
            log.error("level %d: cannot decode level string: %s", level_id, e)
            self.store.mark(level_id, "error", f"decode: {e}"[:500])
            return
        self.store.save_download(level_id, meta, blob, data_len)
        self.downloaded_now += 1

    def write_progress(self, search_type: int, page: int, total: int, page_times: list[float]) -> None:
        counts = self.store.counts()
        pages_total = (total + 9) // 10
        recent = page_times[-50:]
        sec_per_page = sum(recent) / len(recent) if recent else 0.0
        eta_h = max(pages_total - page, 0) * sec_per_page / 3600
        db_mb = self.store.path.stat().st_size / 2**20
        progress = {
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "search_type": search_type,
            "page": page,
            "pages_total": pages_total,
            "levels_total": total,
            "counts": counts,
            "db_mb": round(db_mb, 1),
            "requests": self.client.n_requests,
            "downloaded_this_run": self.downloaded_now,
            "eta_hours": round(eta_h, 1),
        }
        tmp = PROGRESS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(progress, indent=2))
        tmp.replace(PROGRESS_FILE)
        log.info(
            "page %d/%d | downloaded %d skipped %d missing %d error %d | %.0f MB | %d req | ETA %.1f h",
            page, pages_total, counts.get("downloaded", 0), counts.get("skipped", 0), counts.get("missing", 0),
            counts.get("error", 0), db_mb, self.client.n_requests, eta_h,
        )

    def run(self, search_type: int, start_page: int | None, skip_platformer: bool, max_pages: int | None) -> None:
        state_key = f"next_page:{search_type}"
        page = start_page if start_page is not None else int(self.store.get_state(state_key, "0"))
        log.info("crawling listing type=%d from page %d (delay %.1fs)", search_type, page, self.client.delay)
        page_times: list[float] = []
        pages_done = 0
        total = 0
        while True:
            t0 = time.monotonic()
            result = self.call(self.client.search_levels, search_type, page)
            total = result.total or total
            if not result.levels:
                log.info("listing exhausted at page %d", page)
                self.store.set_state(f"listing_done:{search_type}", page)
                break
            for level_id in self.store.add_listing(result.levels, result.creators, result.songs, skip_platformer):
                self.download_one(level_id)
            page += 1
            pages_done += 1
            self.store.set_state(state_key, page)
            page_times.append(time.monotonic() - t0)
            self.write_progress(search_type, page, total, page_times)
            if max_pages is not None and pages_done >= max_pages:
                return
        retry = self.store.retry_candidates(max_attempts=3)
        log.info("retry pass over %d levels", len(retry))
        for level_id in retry:
            self.download_one(level_id)
        self.write_progress(search_type, page, total, page_times)
        log.info("done")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DATA / "levels.sqlite"))
    ap.add_argument("--type", type=int, default=SEARCH_FEATURED, help="getGJLevels21 search type (6 = featured)")
    ap.add_argument("--delay", type=float, default=2.0, help="minimum seconds between requests")
    ap.add_argument("--start-page", type=int, default=None, help="override the saved resume page")
    ap.add_argument("--max-pages", type=int, default=None, help="stop after N listing pages (for testing)")
    ap.add_argument("--include-platformer", action="store_true", help="also download platformer levels")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    DATA.mkdir(parents=True, exist_ok=True)
    if PID_FILE.exists():
        old = PID_FILE.read_text().strip()
        if old.isdigit() and _pid_alive(int(old)):
            log.error("crawler already running (pid %s)", old)
            return 1
    PID_FILE.write_text(str(os.getpid()))

    store = Store(args.db)
    crawler = Crawler(store, GDClient(delay=args.delay))
    signal.signal(signal.SIGTERM, crawler.request_stop)
    signal.signal(signal.SIGINT, crawler.request_stop)
    try:
        crawler.run(args.type, args.start_page, not args.include_platformer, args.max_pages)
    except Stopped:
        log.info("stopped cleanly; rerun to resume")
    finally:
        store.close()
        PID_FILE.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
