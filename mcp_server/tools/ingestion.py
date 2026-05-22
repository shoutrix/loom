"""
Ingestion-related MCP tools.

`ingest_paper` runs loom's existing read-and-ingest pipeline (parse, chunk,
enrich, embed, propositions, entity/relationship extraction, resolution,
community summaries) using MCP sampling for every LLM step. The ingestion
worker queue is bypassed -- ingestion runs foreground for one tool call.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP, Context

from loom.llm.mcp_sampling import MCPSamplingLLMProvider as MCPReasoningProvider
from loom.mcp_server.state import MCPState
from loom.mcp_server.workspace import MCPWorkspaceLoader
from loom.permissions import enforce


def register(mcp: FastMCP, state: MCPState, loader: MCPWorkspaceLoader) -> None:

    @mcp.tool()
    async def ingest_paper(
        ctx: Context,
        workspace_id: str,
        identifier: str,
    ) -> dict[str, Any]:
        """
        Ingest one paper into a workspace's knowledge graph.

        Args:
            workspace_id: target workspace
            identifier: arXiv ID (e.g. "2401.12345"), DOI, "s2:<id>", or URL

        Runs in the foreground (~30 LLM round-trips per paper). Updates the
        workspace's graph + indexes + paper registry on completion. Returns
        {ok, paper_id, doc_id, title, source_type, stats}.

        IMPORTANT — workflow note for the calling agent:
        After this returns successfully, you SHOULD immediately call
        `submit_paper_card(workspace_id, paper_id, card)` with a structured
        13-field review of the paper you just read. You already have the
        paper in context, so producing the card is cheap. Doing it now
        avoids a second LLM call later (loom will otherwise re-extract
        the card via Gemini when the user opens the paper in the UI).

        See `submit_paper_card` for the required card schema.
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        loop = asyncio.get_running_loop()
        llm = MCPReasoningProvider(ctx, loop=loop)
        pipeline = loader.make_pipeline(workspace_id, llm)
        ws = loader.load(workspace_id)

        await ctx.info(f"Ingesting {identifier} into '{workspace_id}'")

        # Mark queued in registry (or upsert manual record)
        paper_id = ws.registry.register_and_queue(identifier)
        ws.registry.save()
        ws.registry.set_status(paper_id, "ingesting")

        from loom.tools.paper_read import read_and_ingest_paper

        result = await asyncio.to_thread(read_and_ingest_paper, pipeline, identifier)

        if result.error:
            ws.registry.set_status(paper_id, "failed", error=result.error)
            ws.registry.save()
            await ctx.info(f"Ingest FAILED: {result.error}")
            return {
                "ok": False,
                "paper_id": paper_id,
                "error": result.error,
            }

        ws.registry.set_status(
            paper_id, "ingested",
            doc_id=result.doc_id,
            ingested_at=datetime.datetime.now().isoformat(),
        )
        loader.save(workspace_id)

        ing = result.ingestion_result
        return {
            "ok": True,
            "paper_id": paper_id,
            "doc_id": result.doc_id,
            "title": result.title,
            "source_type": result.source_type,
            "stats": {
                "num_chunks": getattr(ing, "num_chunks", 0),
                "num_propositions": getattr(ing, "num_propositions", 0),
                "num_entities": getattr(ing, "num_entities", 0),
                "num_relationships": getattr(ing, "num_relationships", 0),
            } if ing else {},
        }

    @mcp.tool()
    async def ingest_papers(
        ctx: Context,
        workspace_id: str,
        identifiers: list[str],
    ) -> dict[str, Any]:
        """
        Ingest multiple papers sequentially. Returns per-paper outcome.

        Foreground only -- expect a multi-minute Claude session per ~5 papers.
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        loop = asyncio.get_running_loop()
        llm = MCPReasoningProvider(ctx, loop=loop)
        pipeline = loader.make_pipeline(workspace_id, llm)
        ws = loader.load(workspace_id)

        from loom.tools.paper_read import read_and_ingest_paper

        outcomes: list[dict[str, Any]] = []
        for i, identifier in enumerate(identifiers, 1):
            await ctx.info(f"[{i}/{len(identifiers)}] Ingesting {identifier}")
            paper_id = ws.registry.register_and_queue(identifier)
            ws.registry.set_status(paper_id, "ingesting")
            try:
                result = await asyncio.to_thread(read_and_ingest_paper, pipeline, identifier)
                if result.error:
                    ws.registry.set_status(paper_id, "failed", error=result.error)
                    outcomes.append({"identifier": identifier, "ok": False, "error": result.error})
                else:
                    ws.registry.set_status(
                        paper_id, "ingested",
                        doc_id=result.doc_id,
                        ingested_at=datetime.datetime.now().isoformat(),
                    )
                    outcomes.append({
                        "identifier": identifier,
                        "ok": True,
                        "doc_id": result.doc_id,
                        "title": result.title,
                    })
            except Exception as e:
                ws.registry.set_status(paper_id, "failed", error=str(e))
                outcomes.append({"identifier": identifier, "ok": False, "error": str(e)})

        ws.registry.save()
        loader.save(workspace_id)
        return {
            "total": len(identifiers),
            "succeeded": sum(1 for o in outcomes if o["ok"]),
            "failed": sum(1 for o in outcomes if not o["ok"]),
            "outcomes": outcomes,
        }

    @mcp.tool()
    async def chat_query(
        ctx: Context,
        workspace_id: str,
        question: str,
    ) -> dict[str, Any]:
        """
        Ask a question over a workspace's ingested corpus.

        Uses loom's hybrid search (semantic + BM25 + graph) to retrieve context,
        then the calling LLM (Claude) composes the answer via MCP sampling.
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
