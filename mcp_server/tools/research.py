"""
Research-kind MCP tools.

These tools wrap loom's existing paper_search pipeline so it runs end-to-end
on the server but uses MCP sampling for every LLM step -- no Gemini reasoning.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.fastmcp import FastMCP, Context

from loom.config import get_settings
from loom.llm.mcp_sampling import MCPSamplingLLMProvider as MCPReasoningProvider
from loom.mcp_server.state import MCPState
from loom.permissions import enforce


def register(mcp: FastMCP, state: MCPState) -> None:

    @mcp.tool()
    async def expand_query(query: str, ctx: Context) -> dict[str, Any]:
        """
        Expand a research query into multi-angle search queries.

        Uses MCP sampling: the calling LLM (Claude) generates the queries; no
        Gemini reasoning involved. Returns one entry per angle with format-
        specific query strings (Semantic Scholar, arXiv, OpenAlex).
        """
        loop = asyncio.get_running_loop()
        llm = MCPReasoningProvider(ctx, loop=loop)

        from loom.tools.paper_search.planner import generate_search_plan

        plan = await asyncio.to_thread(generate_search_plan, llm, "pro", query)
        return {
            "queries": [
                {
                    "label": q.label,
                    "semantic_scholar": q.semantic_scholar,
                    "arxiv": q.arxiv,
                    "openalex": q.openalex,
                    "year_from": q.year_from,
                    "year_to": q.year_to,
                }
                for q in plan.queries
            ],
        }

    @mcp.tool()
    async def research_search(
        ctx: Context,
        query: str,
        workspace_id: str | None = None,
        max_results: int = 30,
        enable_graph_expansion: bool = True,
        graph_expansion_depth: int = 2,
        graph_expansion_max_papers: int = 15,
    ) -> dict[str, Any]:
        """
        Run the full paper search pipeline (Scholar-First) end-to-end.

        Every LLM step (query expansion, relevance scoring, re-rank, citation
        seed selection, drift check, root paper judgment, final ordering) is
        routed back to the MCP client via sampling. The server makes only
        embedding calls and external HTTP calls (arXiv, S2, OpenAlex, Serper).

        Returns a dict with `papers`, `root_papers`, `plan`, and `stats`.
        """
        if workspace_id:
            # Side-effect: results may be registered in the workspace's paper
            # registry, so a write check is appropriate when targeted.
            if (err := enforce(workspace_id, write=True)) is not None:
                return err
        settings = get_settings()
        loop = asyncio.get_running_loop()
        llm = MCPReasoningProvider(ctx, loop=loop)
        if workspace_id:
            llm.set_workspace_context(workspace_id)

        from loom.tools.paper_search.tool import search_papers

        progress_cb = _make_progress_cb(ctx, loop)

        result = await asyncio.to_thread(
            search_papers,
            llm,
            query,
            max_results=max_results,
            semantic_scholar_api_key=settings.semantic_scholar_api_key,
            serper_api_key=settings.serper_api_key,
            enable_graph_expansion=enable_graph_expansion,
            graph_expansion_depth=graph_expansion_depth,
            graph_expansion_max_papers=graph_expansion_max_papers,
            progress_cb=progress_cb,
        )

        # If a workspace was named, optionally register the shortlisted papers.
        if workspace_id and result.get("papers") and not result.get("cancelled"):
            registry = state._open_registry(workspace_id)
            if registry is not None:
                added = registry.register_from_search(result["papers"])
                registry.save()
                result["registry_added"] = added

        return result


def _make_progress_cb(ctx: Context, loop: asyncio.AbstractEventLoop):
    """Bridge sync `progress_cb(step, status)` into async `ctx.info(...)`."""

    def cb(step: str, status: str) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                ctx.info(f"[{step}] {status}"), loop
            )
        except Exception:
            pass

    return cb
