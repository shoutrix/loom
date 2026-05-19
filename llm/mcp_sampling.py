"""
MCP-sampling-backed reasoning provider.

Drop-in LLMProvider that forwards every `.generate()` call back to the
calling MCP client (Claude Code / Desktop) via the sampling protocol. The
server holds no completion API key — the client's host LLM does the
reasoning.

Sync/async bridge: `generate()` is sync but `Context.session.create_message()`
is async. We forward via `asyncio.run_coroutine_threadsafe` into the running
event loop that owns the MCP session.

This module is the canonical home; loom/mcp_server/sampling.py is a thin
back-compat shim.

Usage inside an MCP tool:

    @mcp.tool()
    async def my_tool(ctx: Context, ...) -> ...:
        loop = asyncio.get_running_loop()
        llm = MCPSamplingLLMProvider(ctx, loop=loop)
        result = await asyncio.to_thread(some_sync_pipeline, llm, ...)

Or, construct first and bind ctx per-request:

    llm = MCPSamplingLLMProvider.unbound()
    llm.bind_ctx(ctx, asyncio.get_running_loop())
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from mcp.server.fastmcp import Context
from mcp.types import SamplingMessage, TextContent

from loom.llm.provider import LLMResponse, UsageTracker, estimate_tokens


# Default context-window assumption when the MCP client is Claude (Code or
# Desktop). Override via env or by setting `context_window` on the instance.
_DEFAULT_MCP_CONTEXT_WINDOW = int(os.getenv("LOOM_MCP_CONTEXT_WINDOW", "200000"))


class MCPSamplingLLMProvider:
    """LLM-provider-shaped wrapper around MCP sampling."""

    # Note: usage is a class-level annotation so Protocol structural checks pass
    # even before bind_ctx() is called.
    usage: UsageTracker

    def __init__(
        self,
        ctx: Context | None = None,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        request_timeout: float = 600.0,
        context_window: int = _DEFAULT_MCP_CONTEXT_WINDOW,
    ) -> None:
        self._ctx = ctx
        self._loop = loop
        self._timeout = request_timeout
        self._context_window = context_window
        self.usage = UsageTracker()
        self._workspace_id = "default"

    @classmethod
    def unbound(cls, *, request_timeout: float = 600.0) -> "MCPSamplingLLMProvider":
        """Build an instance whose ctx must be set later via bind_ctx()."""
        return cls(ctx=None, loop=None, request_timeout=request_timeout)

    def bind_ctx(
        self,
        ctx: Context,
        loop: asyncio.AbstractEventLoop,
    ) -> "MCPSamplingLLMProvider":
        """Activate the provider for one MCP request scope."""
        self._ctx = ctx
        self._loop = loop
        return self

    # ----- LLMProvider protocol -----

    @property
    def context_window(self) -> int:
        return self._context_window

    @context_window.setter
    def context_window(self, value: int) -> None:
        self._context_window = int(value)

    def resolve_model_id(self, role: str) -> str:
        # The MCP host picks the model; we surface a stable label for logging.
        return "claude-via-mcp"

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
        """Synchronous generate -- forwards to MCP sampling on the event loop."""
        if self._ctx is None or self._loop is None:
            raise RuntimeError(
                "MCPSamplingLLMProvider is unbound. Call bind_ctx(ctx, loop) "
                "before invoking generate()."
            )
        kwargs: dict[str, Any] = {
            "messages": [
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text=prompt),
                )
            ],
            "max_tokens": max_output_tokens or 4096,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if system_instruction:
            kwargs["system_prompt"] = system_instruction

        coro = self._ctx.session.create_message(**kwargs)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        result = future.result(timeout=self._timeout)

        text = _extract_text(result)
        resp = LLMResponse(
            text=text.strip(),
            model=getattr(result, "model", "claude-via-mcp"),
            estimated_input_tokens=estimate_tokens(prompt),
            estimated_output_tokens=estimate_tokens(text),
        )
        self.usage.record(resp)
        return resp


def _extract_text(result: Any) -> str:
    """Pull text out of a sampling result, tolerating SDK shape variations."""
    content = getattr(result, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            return str(first.get("text", ""))
        return str(getattr(first, "text", ""))
    return ""


# Back-compat alias for code that imported MCPReasoningProvider.
MCPReasoningProvider = MCPSamplingLLMProvider
