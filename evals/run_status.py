"""Exit codes and stop rules shared by the evaluation harnesses and the run_evals_*.py runners.

Every harness saves each finished query to disk immediately, so stopping at any point loses at most the
query in flight. A harness ends with one of these exit codes:

  EXIT_DONE      every selected query has a saved, valid result
  EXIT_PARTIAL   the pass finished, but some queries failed for a non-quota reason; re-running retries them
  EXIT_STOPPED   the LLM is unusable right now (daily quota on every key, billing, outage); stop the whole
                 run and re-run the same command later, e.g. the next day when the quotas reset
"""
EXIT_DONE = 0
EXIT_PARTIAL = 2
EXIT_STOPPED = 75          # EX_TEMPFAIL

# A failed query is retried on the next run. This many failures in a row means every key is failing
# (quota, billing, outage), so the harness stops instead of burning through the remaining queries.
MAX_CONSECUTIVE_FAILURES = 3

# Note: the ordinary per-minute 429 text says "check your plan and billing details", so "billing" alone is not
# a marker; per-minute limits are waited out by the retry logic instead.
_QUOTA_MARKERS = ("perday", "per day", "daily quota", "prepayment credits", "free tier is not available",
                  "permission_denied", "api_key_invalid", "unauthenticated", "(tpd)", "(rpd)")


def is_hard_stop(message: str) -> bool:
    """True when the error means no further call can succeed today (quota, billing, bad key)."""
    low = (message or "").lower()
    return any(m in low for m in _QUOTA_MARKERS)


class FailureTracker:
    """Counts consecutive failures; `should_stop` after a hard-stop error or too many failures in a row."""

    def __init__(self, limit: int = MAX_CONSECUTIVE_FAILURES):
        self.limit = limit
        self.in_a_row = 0
        self.total = 0
        self.stop_reason = ""

    def ok(self) -> None:
        self.in_a_row = 0

    def failed(self, message: str) -> None:
        self.in_a_row += 1
        self.total += 1
        if is_hard_stop(message):
            self.stop_reason = f"quota/billing error: {message[:200]}"
        elif self.in_a_row >= self.limit:
            self.stop_reason = f"{self.in_a_row} failures in a row (last: {message[:200]})"

    @property
    def should_stop(self) -> bool:
        return bool(self.stop_reason)
