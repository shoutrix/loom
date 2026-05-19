"""
OpenRouter-backed reasoning LLM provider.

Uses OpenRouter's OpenAI-compatible /chat/completions endpoint. The same
client can address Anthropic, Google, OpenAI, Meta, etc. models through
a single API key.

Shares utilities (LLMResponse, UsageTracker, rate limiter, token estimation)
with loom.llm.provider; mirrors GeminiLLMProvider's surface so it is a
drop-in replacement at the LLMProvider Protocol level.
"""

from __future__ import annotations

import datetime
import os
import random
import threading
import time
import uuid
from pathlib import Path

import httpx

from loom.config import LLMSettings, RateLimitSettings
from loom.llm.provider import (
    LLMResponse,
    TokenBucketRateLimiter,
    UsageTracker,
    estimate_tokens,
)

# Retryable upstream statuses; OpenRouter passes provider-side rate limits
# through, so 429 is the dominant case in practice.
_RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}

# Static context-window catalogue. Keep conservative; OpenRouter exposes
# longer "extended" / "cached" windows for some Anthropic models, but
# 200_000 is the safe per-call value to plan retrieval against.
_MODEL_CONTEXT_WINDOW: dict[str, int] = {
    "anthropic/claude-opus-4.7": 200_000,
    "anthropic/claude-opus-4.6": 200_000,
    "anthropic/claude-opus-4.5": 200_000,
    "anthropic/claude-sonnet-4.5": 200_000,
    "anthropic/claude-sonnet-4.6": 200_000,
    "anthropic/claude-haiku-4.5": 200_000,
    "google/gemini-2.5-pro": 1_048_576,
    "google/gemini-2.0-flash": 1_048_576,
    "openai/gpt-4o": 128_000,
    "openai/gpt-4o-mini": 128_000,
    "openai/gpt-4.1": 1_000_000,
    "meta-llama/llama-3.3-70b-instruct": 131_072,
}
_DEFAULT_CONTEXT_WINDOW = 128_000


def _lookup_context_window(model_id: str) -> int:
    """Return a known context window or a safe default."""
    return _MODEL_CONTEXT_WINDOW.get(model_id, _DEFAULT_CONTEXT_WINDOW)


class OpenRouterLLMProvider:
    """Reasoning LLM backed by OpenRouter's OpenAI-compatible API."""

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

        api_key = os.getenv("OPENROUTER_API_KEY") or llm_config.openrouter_api_key
        if not api_key:
            raise EnvironmentError(
                "Set the OPENROUTER_API_KEY environment variable before running "
                "Loom with LOOM_LLM_PROVIDER=openrouter."
            )
        self._api_key = api_key
        self._base_url = llm_config.openrouter_base_url.rstrip("/")
        self._extra_headers: dict[str, str] = {}
        if llm_config.openrouter_http_referer:
            self._extra_headers["HTTP-Referer"] = llm_config.openrouter_http_referer
        if llm_config.openrouter_app_title:
            self._extra_headers["X-Title"] = llm_config.openrouter_app_title

        rate_config = rate_config or RateLimitSettings()
        self._rate_limiter = TokenBucketRateLimiter(rate_config.max_calls_per_minute)
        self._daily_limit = rate_config.max_daily_llm_calls
        self._circuit_breaker_threshold = rate_config.circuit_breaker_threshold
        self._circuit_breaker_pause = rate_config.circuit_breaker_pause_seconds
        self._consecutive_failures = 0

        self._workspace_id = workspace_id or "default"
        self._storage_root_dir = Path(
            storage_root_dir or os.getenv("LOOM_STORAGE_ROOT_DIR", ".")
        ).expanduser()
        self._logs_base_dir = self._storage_root_dir / "logs"
        self._log_lock = threading.Lock()
        self._workspace_log_file_by_id: dict[str, Path] = {}

        self._http = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))

    # ----- LLMProvider protocol -----

    @property
    def context_window(self) -> int:
        # Use pro-role context window; chat uses model="pro".
        return _lookup_context_window(self.resolve_model_id("pro"))

    def resolve_model_id(self, role: str) -> str:
        if role == "pro":
            return self.config.openrouter_pro_model
        if role == "flash":
            return self.config.openrouter_flash_model
        return role

    def set_workspace_context(self, workspace_id: str) -> None:
        self._workspace_id = workspace_id or "default"

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

        resolved_model = self.resolve_model_id(model)
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

    # ----- internals -----

    def _resolve_defaults(self, model: str) -> tuple[float, int]:
        if model == "pro" or model == self.config.openrouter_pro_model:
            return self.config.temperature_pro, self.config.max_output_tokens_pro
        return self.config.temperature_flash, self.config.max_output_tokens_flash

    def _build_payload(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
        system_instruction: str | None,
    ) -> dict:
        messages: list[dict] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)
        return headers

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
        url = f"{self._base_url}/chat/completions"
        payload = self._build_payload(
            prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
        )
        headers = self._build_headers()

        for attempt in range(cfg.retry_max_attempts):
            if self._consecutive_failures >= self._circuit_breaker_threshold:
                print(
                    f"  [Circuit breaker] {self._consecutive_failures} consecutive failures. "
                    f"Pausing {self._circuit_breaker_pause}s...", flush=True
                )
                time.sleep(self._circuit_breaker_pause)
                self._consecutive_failures = 0

            try:
                response = self._http.post(url, headers=headers, json=payload)
            except httpx.HTTPError as e:
                self._consecutive_failures += 1
                last_err = e
                self._maybe_sleep_backoff(attempt, f"network error: {e}")
                continue

            if response.status_code in _RETRYABLE_STATUS_CODES:
                self._consecutive_failures += 1
                last_err = RuntimeError(
                    f"OpenRouter HTTP {response.status_code}: {response.text[:300]}"
                )
                self._maybe_sleep_backoff(
                    attempt, f"HTTP {response.status_code}: {response.text[:200]}"
                )
                continue

            if response.status_code >= 400:
                # Non-retryable error
                raise RuntimeError(
                    f"OpenRouter HTTP {response.status_code}: {response.text[:500]}"
                )

            try:
                data = response.json()
            except Exception as e:
                self._consecutive_failures += 1
                last_err = e
                self._maybe_sleep_backoff(attempt, f"json decode: {e}")
                continue

            text = self._extract_text(data)
            if text is None:
                self._consecutive_failures += 1
                last_err = RuntimeError(f"OpenRouter returned no text: {data}")
                self._maybe_sleep_backoff(attempt, "no text in response")
                continue
            return text

        raise RuntimeError(
            f"OpenRouter call failed after {cfg.retry_max_attempts} attempts: {last_err}"
        )

    def _maybe_sleep_backoff(self, attempt: int, reason: str) -> None:
        if attempt >= self.config.retry_max_attempts - 1:
            return
        delay = min(self.config.retry_base_delay * (2 ** attempt), self.config.retry_max_delay)
        jitter = random.uniform(0, delay * 0.2)
        print(
            f"  [OpenRouter retry {attempt + 1}/{self.config.retry_max_attempts}] "
            f"{reason} -- waiting {delay + jitter:.1f}s",
            flush=True,
        )
        time.sleep(delay + jitter)

    @staticmethod
    def _extract_text(data: dict) -> str | None:
        try:
            choices = data.get("choices", [])
            if not choices:
                return None
            first = choices[0]
            message = first.get("message", {}) if isinstance(first, dict) else {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for p in content:
                    if isinstance(p, dict):
                        t = p.get("text")
                        if isinstance(t, str):
                            parts.append(t)
                return "".join(parts) if parts else None
        except Exception:
            return None
        return None

    # ----- logging helpers (mirror GeminiLLMProvider) -----

    def _safe_workspace_name(self, workspace_id: str) -> str:
        safe = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in workspace_id.strip()
        )
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
                f"provider: openrouter",
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
            pass
