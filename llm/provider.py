"""
Shared LLM utilities + back-compat re-export.

Concrete provider implementations now live in sibling modules:
- loom.llm.gemini.GeminiLLMProvider
- (future) loom.llm.openrouter.OpenRouterLLMProvider
- (future) loom.llm.mcp_sampling.MCPSamplingLLMProvider

For new code, prefer:
    from loom.llm.base import LLMProvider           # Protocol for type hints
    from loom.llm import make_llm_provider          # factory

The `LLMProvider` symbol re-exported at the bottom of this module is the
Gemini concrete class, kept here as a back-compat alias for code that still
does `from loom.llm.provider import LLMProvider`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


_RETRYABLE_STATUS_CODES = {429, 500, 503}
CHARS_PER_TOKEN = 4


@dataclass
class LLMResponse:
    text: str
    model: str
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class UsageTracker:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    calls_by_model: dict[str, int] = field(default_factory=dict)

    def record(self, resp: LLMResponse) -> None:
        self.total_input_tokens += resp.estimated_input_tokens
        self.total_output_tokens += resp.estimated_output_tokens
        self.total_calls += 1
        self.calls_by_model[resp.model] = self.calls_by_model.get(resp.model, 0) + 1

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def summary(self) -> str:
        model_str = ", ".join(f"{m}: {c}" for m, c in sorted(self.calls_by_model.items()))
        return (
            f"Calls: {self.total_calls} ({model_str}) | "
            f"Tokens: ~{self.total_tokens:,} "
            f"(in: ~{self.total_input_tokens:,}, out: ~{self.total_output_tokens:,})"
        )


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


class TokenBucketRateLimiter:
    """Thread-safe token-bucket rate limiter."""

    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        self._tokens = float(max_per_minute)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._max, self._tokens + elapsed * (self._max / 60.0))
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            time.sleep(0.5)


# Back-compat re-export. Imported at module bottom to avoid circular import
# (gemini.py imports utilities from this module above).
from loom.llm.gemini import GeminiLLMProvider as LLMProvider  # noqa: E402
