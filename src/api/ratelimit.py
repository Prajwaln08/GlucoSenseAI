"""
Lightweight in-process rate limiting (no extra dependency).

Used as a FastAPI dependency to throttle abuse-prone endpoints — login brute-force and
the public xDRIP push. Keyed by client IP, sliding window.

NOTE: state is per-process (in memory). Fine for a single instance; behind multiple
workers/replicas use a shared store (Redis) — wire that in if you scale out.

(No `from __future__ import annotations` here on purpose: FastAPI must see a real
`Request` type on __call__ to treat it as the request, not a string forward-ref.)
"""

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status


class RateLimiter:
    def __init__(self, times: int, seconds: int) -> None:
        self.times = times
        self.seconds = seconds
        self._hits: dict[str, deque] = defaultdict(deque)

    def __call__(self, request: Request) -> None:
        key = request.client.host if request.client else "anonymous"
        now = time.monotonic()
        window = self._hits[key]
        while window and now - window[0] > self.seconds:
            window.popleft()
        if len(window) >= self.times:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests — please slow down.",
            )
        window.append(now)
