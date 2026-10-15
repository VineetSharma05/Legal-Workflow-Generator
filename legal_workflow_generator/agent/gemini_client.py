"""Gemini client that rotates across several API keys.

GEMINI_API_KEY may hold a comma-separated list of keys. Calls go to the current key; when a call
fails (rate limit, quota, overload, invalid key, ...) the key is put on cooldown and the same call is
retried on the next key. Only when every key has failed is the last error raised, so the callers'
existing retry / LLMUnavailable handling still sees the original Gemini error message.

Drop-in for ``genai.Client``: only ``.models.generate_content`` is used in this project.

    client = RotatingGeminiClient()
    response = client.models.generate_content(model=..., contents=...)
"""
import logging
import re
import threading
import time

from google import genai

from legal_workflow_generator.config.values import GEMINI_API_KEYS

logger = logging.getLogger(__name__)

# How long a failed key is skipped before it is tried again (unless all keys are cooling down).
_DAILY_COOLDOWN = 3600.0
_DEFAULT_COOLDOWN = 60.0
_BAD_KEY_MARKERS = ("PERMISSION_DENIED", "API_KEY_INVALID", "UNAUTHENTICATED")


def _cooldown_for(msg: str) -> float:
    if "PerDay" in msg or "per day" in msg.lower():
        return _DAILY_COOLDOWN
    if any(t in msg for t in _BAD_KEY_MARKERS):   # revoked / invalid / blocked key: won't recover soon
        return _DAILY_COOLDOWN
    m = re.search(r"retry in ([\d.]+)s", msg)
    return float(m.group(1)) + 1 if m else _DEFAULT_COOLDOWN


class KeyPool:
    """Thread-safe list of keys with a sticky current index and per-key cooldowns."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("no Gemini API keys configured")
        self.keys = list(keys)
        self._current = 0
        self._cooling_until = [0.0] * len(self.keys)
        self._lock = threading.Lock()
        self.rotations = 0

    def order(self) -> list[int]:
        """Key indices to try for one call: ready keys first (starting at the current one), then the
        cooling-down keys, soonest-available first."""
        with self._lock:
            now = time.monotonic()
            n = len(self.keys)
            ring = [(self._current + i) % n for i in range(n)]
            ready = [i for i in ring if self._cooling_until[i] <= now]
            cooling = sorted((i for i in ring if self._cooling_until[i] > now), key=lambda i: self._cooling_until[i])
            return ready + cooling

    def mark_failed(self, idx: int, msg: str) -> None:
        with self._lock:
            self._cooling_until[idx] = time.monotonic() + _cooldown_for(msg)
            if self._current == idx:
                self._current = (idx + 1) % len(self.keys)
                self.rotations += 1

    def mark_ok(self, idx: int) -> None:
        with self._lock:
            self._cooling_until[idx] = 0.0
            self._current = idx


class _RotatingModels:
    def __init__(self, owner: "RotatingGeminiClient"):
        self._owner = owner

    def generate_content(self, *args, **kwargs):
        pool = self._owner.pool
        last_exc = None
        for idx in pool.order():
            try:
                response = self._owner._client(idx).models.generate_content(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - any failure moves on to the next key
                last_exc = e
                pool.mark_failed(idx, str(e))
                if len(pool.keys) > 1:
                    logger.warning("Gemini key #%d failed (%s); trying next key", idx + 1, str(e)[:120])
                continue
            pool.mark_ok(idx)
            return response
        assert last_exc is not None
        raise last_exc


class RotatingGeminiClient:
    def __init__(self, keys: list[str] | None = None, pool: KeyPool | None = None):
        self.pool = pool or (KeyPool(keys) if keys is not None else shared_pool())
        self._clients: dict[int, genai.Client] = {}
        self._lock = threading.Lock()
        self.models = _RotatingModels(self)

    def _client(self, idx: int) -> genai.Client:
        with self._lock:
            if idx not in self._clients:
                self._clients[idx] = genai.Client(api_key=self.pool.keys[idx])
            return self._clients[idx]


_shared_pool = None
_shared_pool_lock = threading.Lock()


def shared_pool() -> KeyPool:
    """One pool per process, so a key that hit its limit in one module is skipped by all of them."""
    global _shared_pool
    with _shared_pool_lock:
        if _shared_pool is None:
            _shared_pool = KeyPool(GEMINI_API_KEYS)
        return _shared_pool
