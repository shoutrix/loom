"""
LLM provider with dual-model routing (Pro + Flash).

Wraps the Google GenAI SDK behind a uniform interface with:
- Model routing: generate(prompt, model="pro"|"flash")
- Token-bucket rate limiting
- Circuit breaker
- Retry with exponential backoff + jitter
- Usage tracking per model
"""

from __future__ import annotations

import datetime
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors as genai_errors

from loom.config import LLMSettings, RateLimitSettings

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


class LLMProvider:
    """
    Unified LLM interface with dual-model routing.

    model="pro" -> Gemini 2.5 Pro (deep reasoning)
    model="flash" -> Gemini 2.0 Flash (mechanical tasks)
    """

    def __init__(
        self,
        llm_config: LLMSettings,
        rate_config: RateLimitSettings | None = None,
        *,
        storage_root_dir: str | Path | None = None,
        workspace_id: str = "default",
    ) -> None:
        self.config = llm_config
        self.usage = UsageTracker()

        api_key = os.getenv("GEMINI_API_KEY") or llm_config.gemini_api_key
        if not api_key:
            raise EnvironmentError(
                "Set the GEMINI_API_KEY environment variable before running Loom."
            )

        self._client = genai.Client(api_key=api_key)

        rate_config = rate_config or RateLimitSettings()
        self._rate_limiter = TokenBucketRateLimiter(rate_config.max_calls_per_minute)
        self._daily_limit = rate_config.max_daily_llm_calls
        self._circuit_breaker_threshold = rate_config.circuit_breaker_threshold
        self._circuit_breaker_pause = rate_config.circuit_breaker_pause_seconds
        self._consecutive_failures = 0
        self._workspace_id = workspace_id or "default"
        self._storage_root_dir = Path(storage_root_dir or os.getenv("LOOM_STORAGE_ROOT_DIR", ".")).expanduser()
        self._logs_base_dir = self._storage_root_dir / "logs"
        self._log_lock = threading.Lock()
        self._workspace_log_file_by_id: dict[str, Path] = {}

    def set_workspace_context(self, workspace_id: str) -> None:
        self._workspace_id = workspace_id or "default"

    def _safe_workspace_name(self, workspace_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in workspace_id.strip())
        return safe or "default"

    def _log_file_for_workspace(self, workspace_id: str) -> Path:
        safe_workspace = self._safe_workspace_name(workspace_id)
        with self._log_lock:
            existing = self._workspace_log_file_by_id.get(safe_workspace)
            if existing is not None:
                return existing
            ws_dir = self._logs_base_dir / safe_workspace
            ws_dir.mkdir(parents=True, exist_ok=True)
            log_path = ws_dir / f"{uuid.uuid4()}.txt"
            self._workspace_log_file_by_id[safe_workspace] = log_path
            return log_path

    def _append_llm_log(
        self,
        *,
        workspace_id: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
        prompt: str,
        response_text: str | None,
        error: str | None = None,
        latency_ms: float | None = None,
    ) -> None:
        try:
            log_file = self._log_file_for_workspace(workspace_id)
            ts = datetime.datetime.now(datetime.UTC).isoformat()
            entry = [
                "=" * 80,
                f"timestamp_utc: {ts}",
                f"workspace: {workspace_id}",
                f"model: {model}",
                f"temperature: {temperature}",
                f"max_output_tokens: {max_output_tokens}",
            ]
            if latency_ms is not None:
                entry.append(f"latency_ms: {latency_ms:.2f}")
            if error:
                entry.append(f"error: {error}")
            entry.extend(
                [
                    "prompt:",
                    prompt,
                    "",
                    "raw_response:",
                    response_text if response_text is not None else "",
                    "",
                ]
            )
            with self._log_lock:
                with log_file.open("a", encoding="utf-8") as f:
                    f.write("\n".join(entry))
        except Exception:
            # Never fail a live LLM call because local debug logging failed.
            pass

    def _resolve_model(self, model: str) -> str:
        if model == "pro":
            return self.config.pro_model
        elif model == "flash":
            return self.config.flash_model
        return model

    def _resolve_defaults(self, model: str) -> tuple[float, int]:
        if model == "pro" or model == self.config.pro_model:
            return self.config.temperature_pro, self.config.max_output_tokens_pro
        return self.config.temperature_flash, self.config.max_output_tokens_flash

    def generate(
        self,
        prompt: str,
        *,
        model: str = "flash",
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> LLMResponse:
        if self.usage.total_calls >= self._daily_limit:
            raise RuntimeError(
                f"Daily LLM call limit reached ({self._daily_limit}). "
                f"Increase LOOM_RATE_MAX_DAILY_LLM_CALLS or wait until tomorrow."
            )

        resolved_model = self._resolve_model(model)
        default_temp, default_max_tokens = self._resolve_defaults(model)
        temperature = temperature if temperature is not None else default_temp
        max_tokens = max_output_tokens or default_max_tokens

        self._rate_limiter.acquire()

        t0 = time.time()
        try:
            raw = self._call_with_retry(
                prompt,
                model=resolved_model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction,
            )
        except Exception as e:
            self._append_llm_log(
                workspace_id=self._workspace_id,
                model=resolved_model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                prompt=prompt,
                response_text=None,
                error=str(e),
                latency_ms=(time.time() - t0) * 1000,
            )
            raise
        latency = (time.time() - t0) * 1000
        self._append_llm_log(
            workspace_id=self._workspace_id,
            model=resolved_model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            prompt=prompt,
            response_text=raw,
            latency_ms=latency,
        )

        self._consecutive_failures = 0
        resp = LLMResponse(
            text=raw.strip(),
            model=resolved_model,
            estimated_input_tokens=estimate_tokens(prompt),
            estimated_output_tokens=estimate_tokens(raw),
            latency_ms=latency,
        )
        self.usage.record(resp)
        return resp

    def _call_with_retry(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
        system_instruction: str | None,
    ) -> str:
        cfg = self.config
        last_err: Exception | None = None

        for attempt in range(cfg.retry_max_attempts):
            if self._consecutive_failures >= self._circuit_breaker_threshold:
                print(
                    f"  [Circuit breaker] {self._consecutive_failures} consecutive failures. "
                    f"Pausing {self._circuit_breaker_pause}s...", flush=True
                )
                time.sleep(self._circuit_breaker_pause)
                self._consecutive_failures = 0

            try:
                config_kwargs: dict[str, Any] = {
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                }
                gen_config = genai.types.GenerateContentConfig(**config_kwargs)
                if system_instruction:
                    gen_config.system_instruction = system_instruction

                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=gen_config,
                )

                text = None
                try:
                    text = response.text
                except Exception:
                    pass

                if text is None and response.candidates:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        text = "".join(
                            p.text for p in candidate.content.parts if hasattr(p, "text") and p.text
                        ) or None

                if text is None:
                    block_reason = getattr(response, "prompt_feedback", None)
                    candidates = getattr(response, "candidates", [])
                    finish = candidates[0].finish_reason if candidates else "no_candidates"
                    raise RuntimeError(
                        f"LLM returned None text (finish_reason={finish}, "
                        f"prompt_feedback={block_reason})"
                    )
                return text

            except genai_errors.APIError as e:
                self._consecutive_failures += 1
                status = getattr(e, "code", 0)
                if status not in _RETRYABLE_STATUS_CODES:
                    raise
                last_err = e
                if attempt < cfg.retry_max_attempts - 1:
                    delay = min(cfg.retry_base_delay * (2 ** attempt), cfg.retry_max_delay)
                    jitter = random.uniform(0, delay * 0.2)
                    print(
                        f"  [LLM retry {attempt + 1}/{cfg.retry_max_attempts}] "
                        f"HTTP {status}: {getattr(e, 'message', e)} -- waiting {delay + jitter:.1f}s",
                        flush=True,
                    )
                    time.sleep(delay + jitter)

            except Exception as e:
                self._consecutive_failures += 1
                last_err = e
                if attempt < cfg.retry_max_attempts - 1:
                    delay = min(cfg.retry_base_delay * (2 ** attempt), cfg.retry_max_delay)
                    jitter = random.uniform(0, delay * 0.2)
                    print(
                        f"  [LLM retry {attempt + 1}/{cfg.retry_max_attempts}] "
                        f"{e} -- waiting {delay + jitter:.1f}s",
                        flush=True,
                    )
                    time.sleep(delay + jitter)

        raise RuntimeError(
            f"LLM call failed after {cfg.retry_max_attempts} attempts: {last_err}"
        )
