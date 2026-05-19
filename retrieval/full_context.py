"""
FullContextRetriever — stuff the entire workspace into the model's window.

Built for workspaces that fit in the active model's context (with safety
margin). The adaptive dispatcher (P4 sibling) decides when to invoke this
vs the graph-hybrid path.

The retriever does not filter by query — every chunk, every proposition,
and a compact serialization of the graph are returned. The LLM then has
the whole corpus to reason over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loom.llm.provider import estimate_tokens
from loom.retrieval.base import RetrievalResult

if TYPE_CHECKING:
    from loom.chat.engine import ChatMessage
    from loom.graph.store import GraphStore
    from loom.search.semantic import DualSemanticIndex

# Per-chunk preview cap — full chunk text is still passed but we truncate
# extremely long chunks just to be safe against pathological inputs.
_MAX_CHUNK_CHARS = 50_000


class FullContextRetriever:
    """Return every chunk + proposition + a compact graph dump."""

    name = "full_context"

    def __init__(
        self,
        *,
        semantic_index: "DualSemanticIndex",
        graph: "GraphStore",
    ) -> None:
        self.semantic_index = semantic_index
        self.graph = graph

    def retrieve(
        self,
        query: str,
        *,
        history: "list[ChatMessage] | None" = None,
    ) -> RetrievalResult:
        from loom.search.hybrid import HybridResult

        chunks: list[HybridResult] = []
        chunk_idx = self.semantic_index.chunk_index
        for i, cid in enumerate(chunk_idx.ids):
            text = chunk_idx.texts[i][:_MAX_CHUNK_CHARS]
            chunks.append(
                HybridResult(
                    id=cid,
                    text=text,
                    score=1.0,
                    doc_id=chunk_idx.doc_ids[i] if i < len(chunk_idx.doc_ids) else "",
                    is_proposition=False,
                    source="full_context",
                )
            )

        propositions: list[HybridResult] = []
        prop_idx = self.semantic_index.proposition_index
        for i, pid in enumerate(prop_idx.ids):
            text = prop_idx.texts[i][:_MAX_CHUNK_CHARS]
            propositions.append(
                HybridResult(
                    id=pid,
                    text=text,
                    score=1.0,
                    doc_id=prop_idx.doc_ids[i] if i < len(prop_idx.doc_ids) else "",
                    is_proposition=True,
                    source="full_context",
                )
            )

        graph_context = self._serialize_graph()

        # Token estimate sums everything we'd hand to the LLM.
        token_estimate = estimate_tokens(graph_context)
        for r in chunks:
            token_estimate += estimate_tokens(r.text)
        for r in propositions:
            token_estimate += estimate_tokens(r.text)

        all_results = propositions + chunks  # propositions first, like hybrid

        return RetrievalResult(
            chunks=chunks,
            propositions=propositions,
            graph_context=graph_context,
            all_results=all_results,
            total_tokens_estimated=token_estimate,
            retriever_used=self.name,
        )

    def _serialize_graph(self) -> str:
        """Compact human-readable dump of entities, relationships, communities."""
        g = self.graph
        if not g.entities and not g.communities:
            return ""

        lines: list[str] = []

        if g.communities:
            lines.append("=== COMMUNITIES ===")
            for c in g.communities.values():
                summary = (c.summary or "").strip()
                if summary:
                    lines.append(f"[community-{c.id}] {summary}")

        if g.entities:
            lines.append("\n=== ENTITIES ===")
            for ent in g.entities.values():
                desc = (ent.description or "").strip()
                if ent.name:
                    if desc:
                        lines.append(f"- {ent.name}: {desc}")
                    else:
                        lines.append(f"- {ent.name}")

        if g.relationships:
            lines.append("\n=== RELATIONSHIPS ===")
            for rel in g.relationships:
                src = g.entities.get(rel.source_id)
                tgt = g.entities.get(rel.target_id)
                src_name = src.name if src else rel.source_id
                tgt_name = tgt.name if tgt else rel.target_id
                desc = (rel.description or "").strip()
                if desc:
                    lines.append(f"- {src_name} -> {tgt_name}: {desc}")
                else:
                    lines.append(f"- {src_name} -> {tgt_name}")

        return "\n".join(lines)
