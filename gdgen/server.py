"""Minimal, polite client for the official Geometry Dash servers (boomlings.com)."""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

import requests
from urllib3.util import SKIP_HEADER

BASE_URL = "https://www.boomlings.com/database/"
SECRET = "Wmfd2893gb7"
COMMON_PARAMS = {"gameVersion": "22", "binaryVersion": "42", "secret": SECRET}

# getGJLevels21 `type` values we care about.
SEARCH_FEATURED = 6
SEARCH_AWARDED = 11
SEARCH_HALL_OF_FAME = 16

log = logging.getLogger(__name__)


class RateLimited(Exception):
    """Server or Cloudflare told us to slow down."""


class ServerError(Exception):
    """Unexpected response (HTTP error, malformed body)."""


def parse_kv(text: str, sep: str = ":") -> dict[str, str]:
    parts = text.split(sep)
    return dict(zip(parts[::2], parts[1::2]))


@dataclass
class SearchPage:
    levels: list[dict[str, str]]
    creators: dict[int, dict[str, str]] = field(default_factory=dict)
    songs: list[dict[str, str]] = field(default_factory=list)
    total: int = 0
    offset: int = 0
    amount: int = 0


def parse_search_response(text: str) -> SearchPage:
    text = text.strip()
    if text in ("-1", ""):
        return SearchPage(levels=[])
    sections = text.split("#")
    if len(sections) < 4:
        raise ServerError(f"malformed search response: {text[:200]!r}")
    levels = [parse_kv(x) for x in sections[0].split("|") if x]
    creators = {}
    for entry in sections[1].split("|"):
        fields = entry.split(":")
        if len(fields) >= 3 and fields[0].isdigit():
            creators[int(fields[0])] = {"name": fields[1], "account_id": fields[2]}
    songs = [parse_kv(s, "~|~") for s in sections[2].split("~:~") if s]
    total, offset, amount = (int(x) for x in sections[3].split(":")[:3])
    return SearchPage(levels, creators, songs, total, offset, amount)


def parse_download_response(text: str) -> dict[str, str] | None:
    """Returns level fields (key '4' is the level string) or None if the level doesn't exist."""
    text = text.strip()
    if text == "-1":
        return None
    level = parse_kv(text.split("#")[0])
    if "4" not in level or "1" not in level:
        raise ServerError(f"malformed download response: {text[:200]!r}")
    return level


class GDClient:
    """Enforces a minimum gap between requests; raises RateLimited/ServerError for the caller to back off."""

    def __init__(self, delay: float = 2.0, jitter: float = 0.5, timeout: float = 60.0):
        self.delay = delay
        self.jitter = jitter
        self.timeout = timeout
        self.session = requests.Session()
        # Cloudflare in front of the GD servers 403s urllib3's default User-Agent; the game sends none.
        self.session.headers["User-Agent"] = SKIP_HEADER
        self._last_request = 0.0
        self.n_requests = 0

    def _throttle(self) -> None:
        gap = self.delay + random.uniform(0, self.jitter)
        elapsed = time.monotonic() - self._last_request
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_request = time.monotonic()

    def post(self, endpoint: str, **params) -> str:
        self._throttle()
        self.n_requests += 1
        try:
            resp = self.session.post(BASE_URL + endpoint, data={**COMMON_PARAMS, **params}, timeout=self.timeout)
        except requests.RequestException as e:
            raise ServerError(f"{type(e).__name__}: {e}") from e
        text = resp.text
        if resp.status_code == 429 or "error code: 1015" in text or text.lstrip().startswith("<"):
            # Rate limit or a Cloudflare challenge/block page instead of game data.
            raise RateLimited(f"HTTP {resp.status_code}: {text[:120]!r}")
        if resp.status_code != 200:
            raise ServerError(f"HTTP {resp.status_code}: {text[:200]!r}")
        return text

    def search_levels(self, search_type: int, page: int, **filters) -> SearchPage:
        return parse_search_response(self.post("getGJLevels21.php", type=search_type, page=page, str="", **filters))

    def download_level(self, level_id: int) -> dict[str, str] | None:
        level = parse_download_response(self.post("downloadGJLevel22.php", levelID=level_id))
        if level is not None and level["1"] != str(level_id):
            raise ServerError(f"asked for {level_id}, got {level['1']}")
        return level
