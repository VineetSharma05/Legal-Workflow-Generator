"""Groq chat completion, used by every LLM call when the run's provider is "groq".

Groq's free tier limits tokens per minute (TPM, 8K for gpt-oss-120b) more tightly than requests, so calls
are paced by a token budget: each call reserves an estimate before it is sent and the reservation is
replaced by the real usage afterwards. A 429 on the per-minute limit is waited out here; a 429 on the
per-day limit (TPD/RPD) is raised so the caller can stop the run.
"""
import os
import re
import threading
import time
from collections import deque

import legal_workflow_generator.config.values as app_config

_client = None
_lock = threading.Lock()

# Running token totals for this process (read by the eval harnesses to report cost).
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# gpt-oss models accept a reasoning effort (low | medium | high). Empty = provider default.
REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "").strip().lower()
# Fixed sampling temperature for every call that does not pass its own; empty = provider default.
TEMPERATURE = os.getenv("GROQ_TEMPERATURE", "").strip()
# Token budget per rolling minute (keep a little under the account's TPM limit).
TPM_BUDGET = int(os.getenv("GROQ_TPM", "7600"))
# Expected completion tokens (reasoning + answer) reserved per call before the real figure is known.
COMPLETION_RESERVE = int(os.getenv("GROQ_COMPLETION_RESERVE", "900"))


class GroqDailyLimit(RuntimeError):
    """The per-day token or request quota is used up. Message contains 'per day' on purpose."""


_window: deque = deque()          # (timestamp, tokens) of calls in the last 60 s
_window_lock = threading.Lock()


def _window_total(now: float) -> int:
    while _window and now - _window[0][0] >= 60.0:
        _window.popleft()
    return sum(t for _, t in _window)


def _reserve(tokens: int) -> list:
    """Block until `tokens` fit in the rolling-minute budget, then record them. Returns the entry."""
    tokens = min(tokens, TPM_BUDGET)
    while True:
        with _window_lock:
            now = time.monotonic()
            used = _window_total(now)
            if used + tokens <= TPM_BUDGET or not _window:
                entry = [now, tokens]
                _window.append(entry)
                return entry
            wait = 60.0 - (now - _window[0][0]) + 0.05
        time.sleep(max(wait, 0.05))


def _settle(entry: list, actual: int) -> None:
    with _window_lock:
        entry[1] = actual
        # deque holds the list object itself, so the update is visible to _window_total
        for i, e in enumerate(_window):
            if e is entry:
                _window[i] = (e[0], actual)
                break


def _get_client():
    global _client
    with _lock:
        if _client is None:
            if not app_config.GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file to use the groq provider.")
            from groq import Groq
            # Retries are handled here (TPM) and by the callers; the SDK's own retries would hide 429s.
            _client = Groq(api_key=app_config.GROQ_API_KEY, max_retries=0, timeout=120)
    return _client


def _record(usage) -> int:
    if usage is None:
        return 0
    total = int(getattr(usage, "total_tokens", 0) or 0)
    with _lock:
        USAGE["calls"] += 1
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            USAGE[k] += int(getattr(usage, k, 0) or 0)
    return total


_WAIT = re.compile(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)")


def _retry_after(msg: str) -> float:
    m = _WAIT.search(msg)
    if not m:
        return 5.0
    minutes = float(m.group(1) or 0)
    val = float(m.group(2))
    return minutes * 60 + (val / 1000 if m.group(3) == "ms" else val)


def complete(system: str, user: str, temperature: float | None = None, json_mode: bool = False,
             max_attempts: int = 8) -> str:
    messages: list = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    kwargs = {"model": app_config.GROQ_MODEL, "messages": messages}
    if temperature is None and TEMPERATURE:
        temperature = float(TEMPERATURE)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if REASONING_EFFORT and "gpt-oss" in app_config.GROQ_MODEL:
        kwargs["reasoning_effort"] = REASONING_EFFORT
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    estimate = (len(system) + len(user)) // 3 + COMPLETION_RESERVE
    for attempt in range(max_attempts):
        entry = _reserve(estimate)
        try:
            r = _get_client().chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            _settle(entry, estimate)        # a rejected call still counts against the minute on Groq's side
            msg = str(e)
            low = msg.lower()
            if "per day" in low or "(tpd)" in low or "(rpd)" in low:
                raise GroqDailyLimit(f"groq daily quota exhausted (per day): {msg[:200]}") from e
            transient = ("429" in msg or "rate_limit" in low or "503" in msg or "500" in msg or "502" in msg
                         or "timeout" in low or "timed out" in low or "connection" in low or "over capacity" in low)
            if transient and attempt < max_attempts - 1:
                time.sleep(min(_retry_after(msg) + 0.5, 65))
                continue
            raise
        _settle(entry, _record(getattr(r, "usage", None)) or estimate)
        return (r.choices[0].message.content or "").strip()
    raise RuntimeError("unreachable")
