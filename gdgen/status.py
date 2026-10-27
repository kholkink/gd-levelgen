"""Prints crawler progress without touching the database (safe while the crawler runs)."""
from __future__ import annotations

import json
import sys

from .crawl import DATA, PID_FILE, PROGRESS_FILE, _pid_alive


def main() -> int:
    pid = PID_FILE.read_text().strip() if PID_FILE.exists() else ""
    running = pid.isdigit() and _pid_alive(int(pid))
    print(f"crawler: {'RUNNING (pid ' + pid + ')' if running else 'not running'}")
    if PROGRESS_FILE.exists():
        p = json.loads(PROGRESS_FILE.read_text())
        c = p["counts"]
        print(f"updated:  {p['updated']}")
        print(f"pages:    {p['page']}/{p['pages_total']}  (~{p['levels_total']} levels in listing)")
        print(f"levels:   downloaded {c.get('downloaded', 0)}, skipped(platformer) {c.get('skipped', 0)}, "
              f"missing {c.get('missing', 0)}, error {c.get('error', 0)}")
        print(f"db size:  {p['db_mb']} MB")
        print(f"ETA:      {p['eta_hours']} h")
    log = DATA / "crawl.log"
    if log.exists():
        print("--- last log lines ---")
        print("\n".join(log.read_text(errors="replace").splitlines()[-5:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
