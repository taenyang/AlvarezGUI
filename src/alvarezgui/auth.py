from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import defaultdict, deque

SESSION_COOKIE = "alvarez_session"
SESSION_SECONDS = 12 * 3600
MAX_FAILURES = 5
FAILURE_WINDOW_SECONDS = 300


class ConsoleAuth:
    """Single shared password; browsers get a signed session cookie, programs send a Bearer header."""

    def __init__(self, password: str) -> None:
        if not password:
            raise ValueError("console password must not be empty")
        self._password = password.encode()
        # Per-process key: restarting the console logs everyone out, which is acceptable for one admin password.
        self._key = secrets.token_bytes(32)
        self._failures: dict[str, deque[float]] = defaultdict(deque)

    def password_matches(self, candidate: str) -> bool:
        return hmac.compare_digest(candidate.encode(), self._password)

    def bearer_ok(self, authorization: str | None) -> bool:
        if not authorization or not authorization.startswith("Bearer "):
            return False
        return self.password_matches(authorization.removeprefix("Bearer "))

    def issue_session(self, now: float | None = None) -> str:
        expires = int((now or time.time()) + SESSION_SECONDS)
        return f"{expires}.{self._sign(str(expires))}"

    def session_ok(self, cookie: str | None, now: float | None = None) -> bool:
        if not cookie or "." not in cookie:
            return False
        expires, signature = cookie.split(".", 1)
        if not expires.isdigit() or not hmac.compare_digest(signature, self._sign(expires)):
            return False
        return int(expires) > (now or time.time())

    def login(self, client_ip: str, candidate: str, now: float | None = None) -> bool | None:
        """Returns True on success, False on a wrong password, None while the client is rate limited."""
        now = now or time.time()
        failures = self._failures[client_ip]
        while failures and failures[0] < now - FAILURE_WINDOW_SECONDS:
            failures.popleft()
        if len(failures) >= MAX_FAILURES:
            return None
        if self.password_matches(candidate):
            failures.clear()
            return True
        failures.append(now)
        return False

    def _sign(self, value: str) -> str:
        return hmac.new(self._key, value.encode(), hashlib.sha256).hexdigest()
