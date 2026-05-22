"""
Chat / query MCP tools.

`chat_query` runs hybrid retrieval against the workspace and uses MCP
sampling to compose the answer — the calling agent's LLM does the
reasoning. This is the one tool in the MCP server that intentionally
keeps the per-request MCP-sampling pattern, because the agent IS the
consumer of the answer and naturally pays the reasoning cost.

The previous `ingest_paper` and `ingest_papers` tools were removed in
favor of the fire-and-forget `submit_paper(s)` / `submit_paper_card(s)`
tools — see paper_card_tools.py. Ingestion is now loom-internal and
runs in the background using loom's configured server-side LLM.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from loom.llm.mcp_sampling import MCPSamplingLLMProvider as MCPReasoningProvider
from loom.mcp_server.state import MCPState
from loom.mcp_server.workspace import MCPWorkspaceLoader
from loom.permissions import enforce


def register(mcp: FastMCP, state: MCPState, loader: MCPWorkspaceLoader) -> None:

    @mcp.tool()
    async def chat_query(
        ctx: Context,
        workspace_id: str,
        question: str,
    ) -> dict[str, Any]:
        """
        Ask a question over a workspace's ingested corpus.

        Uses loom's hybrid retrieval (semantic + BM25 + graph context, or
        adaptive full-context when the workspace fits in your context
        window) to fetch relevant passages, then the calling LLM (you,
        the agent) composes the answer via MCP sampling.

        Args:
            workspace_id: target workspace (must be readable for this
                subscriber).
            question: the natural-language question.

        Returns:
            {answer, sources, graph_context, num_chunks_retrieved,
             num_propositions_retrieved}
        """
        if (err := enforce(workspace_id, write=False)) is not None:
            return err
        loop = asyncio.get_running_loop()
        llm = MCPReasoningProvider(ctx, loop=loop)
        engine = loader.make_chat_engine(workspace_id, llm)

        result = await asyncio.to_thread(engine.chat, question)
        ws = loader.load(workspace_id)
        chat_path = ws.settings.data_dir / "chat_history.json"
        engine.save_history(chat_path)
        return {
            "answer": result.answer,
            "sources": result.sources,
            "graph_context": result.graph_context,
            "num_chunks_retrieved": result.num_chunks_retrieved,
            "num_propositions_retrieved": result.num_propositions_retrieved,
        }
