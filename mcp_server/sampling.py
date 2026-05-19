"""
MCP-sampling-based reasoning provider.

Drop-in replacement for `loom.llm.provider.LLMProvider` that routes every
`.generate()` call back to the MCP client (Claude Code/Desktop) via the
sampling protocol. The server holds no completion API key -- the client's LLM
does the reasoning.

Sync/async bridge: `LLMProvider.generate()` is a sync method but
`Context.session.create_message()` is async. We use
`asyncio.run_coroutine_threadsafe` to forward sync calls into the running
event loop where the MCP session lives.

Usage in an MCP tool:

    @mcp.tool()
    async def my_tool(ctx: Context, ...) -> ...:
        loop = asyncio.get_running_loop()
        llm = MCPReasoningProvider(ctx, loop=loop)
        # pass `llm` anywhere a `LLMProvider` was previously expected
        result = await asyncio.to_thread(some_sync_pipeline, llm, ...)
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.fastmcp import Context
from mcp.types import SamplingMessage, TextContent

from loom.llm.provider import LLMResponse, UsageTracker, estimate_tokens


class MCPReasoningProvider:
    """LLM-provider-shaped wrapper around MCP sampling."""

    def __init__(
        self,
        ctx: Context,
        *,
        loop: asyncio.AbstractEventLoop,
        request_timeout: float = 600.0,
    ) -> None:
        self._ctx = ctx
        self._loop = loop
        self._timeout = request_timeout
        self.usage = UsageTracker()
        self._workspace_id = "default"

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
