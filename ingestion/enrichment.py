"""
Contextual chunk enrichment (Anthropic's technique).

One Flash call reads the document title + abstract and generates a
2-sentence context prefix. Each chunk is prepended with this context
before embedding, anchoring it to the document's topic.

Result: 49% reduction in retrieval failures (Anthropic benchmark).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loom.llm.provider import LLMProvider
    from loom.ingestion.chunker import Chunk

ENRICHMENT_PROMPT = """You are helping prepare document chunks for a semantic search index.

Given a document's title and abstract/summary, write exactly 2 sentences that establish the context for any chunk from this document. These sentences will be prepended to each chunk before embedding.

Requirements:
- Mention the document title or key topic
- Establish the domain and main contribution
- Be factual and specific, not generic
- Keep it under 60 words total

Document title: {title}

Abstract/Summary:
{abstract}

Write your 2-sentence context prefix:"""


def generate_context_prefix(
    llm: LLMProvider,
    title: str,
    abstract: str,
) -> str:
    """Generate a 2-sentence context prefix for all chunks in a document."""
    prompt = ENRICHMENT_PROMPT.format(
        title=title,
        abstract=abstract[:1000],
    )
    resp = llm.generate(prompt, model="flash", temperature=0.1, max_output_tokens=150)
    prefix = resp.text.strip()
    if not prefix.endswith("."):
        prefix += "."
    return prefix


def enrich_chunks(chunks: list[Chunk], context_prefix: str) -> list[str]:
    """Prepend context prefix to each chunk's text for embedding.

    Returns a list of enriched text strings (same order as input chunks).
    The original Chunk.text is NOT modified.
    """
    enriched: list[str] = []
    for chunk in chunks:
        enriched_text = f"{context_prefix}\n\n{chunk.text}"
        enriched.append(enriched_text)
    return enriched
