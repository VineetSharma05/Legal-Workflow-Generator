"""Shared rate limits for every LLM call in the app (one budget per provider, per process).

Every Gemini/Groq call must take a slot from LIMITERS[provider] so that parallel domain workers,
the query classifiers and the RAG generator together stay under the free-tier quota.

  * sliding window: at most `rpm` calls in any 60 s window
  * in-flight cap:   at most `LLM_MAX_PARALLEL` calls running at once

Usage:
    with LIMITERS["gemini"].slot():
        response = client.models.generate_content(...)

Limits are per process. Running several server workers multiplies the effective rate, so keep
GEMINI_RPM at (quota / number of workers).
"""
import os
import threading
import time
from collections import deque
from contextlib import contextmanager


class RateLimiter:
    """Sliding-window limiter (at most `rpm` calls per `window` seconds) plus a cap on in-flight calls."""

    def __init__(self, rpm: int, max_parallel: int = 4, window: float = 60.0):
        self.rpm = max(1, int(rpm))
        self.window = float(window)
        self._stamps = deque()
        self._lock = threading.Lock()
        self._sem = threading.BoundedSemaphore(max(1, int(max_parallel)))
        self.calls = 0           # requests actually sent
        self.retries = 0         # requests that had to be retried (429 / 5xx)

    def set_rpm(self, rpm: int) -> None:
        self.rpm = max(1, int(rpm))

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._stamps and now - self._stamps[0] >= self.window:
                    self._stamps.popleft()
                if len(self._stamps) < self.rpm:
                    self._stamps.append(now)
                    self.calls += 1
                    return
                wait = self.window - (now - self._stamps[0]) + 0.05
            time.sleep(max(wait, 0.05))

    @contextmanager
    def slot(self):
        with self._sem:
            self.acquire()
            yield


_MAX_PARALLEL = int(os.getenv("LLM_MAX_PARALLEL", "4"))
LIMITERS = {
    "gemini": RateLimiter(int(os.getenv("GEMINI_RPM", "12")), _MAX_PARALLEL),   # free tier allows 15/min
    "groq": RateLimiter(int(os.getenv("GROQ_RPM", "25")), _MAX_PARALLEL),
}


def set_rate_limit(provider: str, rpm: int) -> None:
    LIMITERS[provider].set_rpm(rpm)
