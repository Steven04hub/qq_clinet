"""Bounded per-user cooldowns shared across all groups in this process."""

import math
import time
from collections import OrderedDict
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        interval_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_users: int = 10000,
    ) -> None:
        if (
            isinstance(interval_seconds, bool)
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be finite and positive")
        if isinstance(max_users, bool) or not isinstance(max_users, int) or max_users < 1:
            raise ValueError("max_users must be a positive integer")
        self.interval_seconds = interval_seconds
        self.max_users = max_users
        self._clock = clock
        self._last_allowed: OrderedDict[int, float] = OrderedDict()

    def allow(self, user_id: int) -> bool:
        """Return immediately; rejected attempts never extend the user's cooldown."""
        now = self._clock()
        # Fixed-duration cooldowns expire in insertion order. Remove only expired
        # entries so a flood of new users cannot evict an active user's cooldown.
        while self._last_allowed:
            oldest_user, last_allowed = next(iter(self._last_allowed.items()))
            if now - last_allowed < self.interval_seconds:
                break
            del self._last_allowed[oldest_user]
        if user_id in self._last_allowed or len(self._last_allowed) >= self.max_users:
            return False
        self._last_allowed[user_id] = now
        return True
