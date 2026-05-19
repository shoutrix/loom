"""
Gemini concrete provider implementations.

GeminiLLMProvider — dual-model reasoning (Pro + Flash) via google-genai SDK.
GeminiEmbeddingProvider — batch embeddings via Gemini embedding model.

Shares utilities (LLMResponse, UsageTracker, rate limiter, token estimation)
with loom.llm.provider.
"""

from __future__ import annotations

import datetime
import os
import random
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from google import genai
from google.genai import errors as genai_errors

from loom.config import LLMSettings, RateLimitSettings
from loom.llm.provider import (
    _RETRYABLE_STATUS_CODES,
    LLMResponse,
    TokenBucketRateLimiter,
    UsageTracker,
    estimate_tokens,
)

# Gemini 2.5 Pro and 2.0 Flash both have ~1M context windows.
_GEMINI_CONTEXT_WINDOW = 1_048_576

# Embedding batch / retry constants
MAX_BATCH_SIZE = 100
EMBED_RETRY_ATTEMPTS = 3
EMBED_RETRY_DELAY = 2.0


class GeminiLLMProvider:
    """Reasoning LLM backed by Google GenAI (Gemini Pro / Flash)."""

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
        self._storage_root_dir = Path(
            storage_root_dir or os.getenv("LOOM_STORAGE_ROOT_DIR", ".")
        ).expanduser()
        self._logs_base_dir = self._storage_root_dir / "logs"
        self._log_lock = threading.Lock()
        self._workspace_log_file_by_id: dict[str, Path] = {}

    @property
    def context_window(self) -> int:
        return _GEMINI_CONTEXT_WINDOW

    def resolve_model_id(self, role: str) -> str:
        return self._resolve_model(role)

    def set_workspace_context(self, workspace_id: str) -> None:
        self._workspace_id = workspace_id or "default"

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
                            p.text for p in candidate.content.parts
                            if hasattr(p, "text") and p.text
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


class GeminiEmbeddingProvider:
    """Batch-oriented embedding provider using a Gemini embedding model."""

    def __init__(self, config: LLMSettings) -> None:
        self.model_name = config.embedding_model
        self.dimensions = config.embedding_dimensions
        # Back-compat alias used by existing callers
        self.model = config.embedding_model
        api_key = os.getenv("GEMINI_API_KEY") or config.gemini_api_key
        if not api_key:
            raise EnvironmentError("Set GEMINI_API_KEY before running Loom.")
        self._client = genai.Client(api_key=api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[start : start + MAX_BATCH_SIZE]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        arr = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return arr / norms

    def embed_single(self, text: str) -> np.ndarray:
        result = self.embed([text])
        return result[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(EMBED_RETRY_ATTEMPTS):
            try:
                result = self._client.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                )
                return [e.values for e in result.embeddings]
            except Exception as e:
                last_err = e
                if attempt < EMBED_RETRY_ATTEMPTS - 1:
                    time.sleep(EMBED_RETRY_DELAY * (2 ** attempt))
        raise RuntimeError(
            f"Embedding failed after {EMBED_RETRY_ATTEMPTS} attempts: {last_err}"
        )
