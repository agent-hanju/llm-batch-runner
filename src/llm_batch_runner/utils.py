from __future__ import annotations

import re
import threading
import time
import urllib.parse


def normalize_base_url(base_url: str) -> str:
    """Normalize a base URL: validate scheme, strip trailing slashes, auto-append /v1."""
    url = base_url.strip().rstrip("/")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"base_url must start with http:// or https://, got: {base_url!r}")
    if not parsed.netloc:
        raise ValueError(f"base_url has no host: {base_url!r}")
    if not re.search(r"/v\d+$", parsed.path):
        url += "/v1"
    return url


class RateLimiter:
    """Token-bucket rate limiter (thread-safe).

    Allows ``rate`` requests per ``period`` seconds. Callers block in
    ``acquire()`` until a token is available.
    """

    def __init__(self, rate: int, period: float = 60.0) -> None:
        self._rate = rate
        self._period = period
        self._tokens: float = rate
        self._last: float = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._rate, self._tokens + elapsed * self._rate / self._period)
            self._last = now
            if self._tokens < 1:
                wait = (1 - self._tokens) * self._period / self._rate
                self._tokens = 0.0
            else:
                wait = 0.0
                self._tokens -= 1
        if wait:
            time.sleep(wait)
