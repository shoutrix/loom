"""
Proposition extraction: decompose chunks into atomic, self-contained facts.

Uses Gemini Pro for deep reasoning. Each proposition is independently
verifiable and embeddable. Serves double duty:
  1. Sharper embeddings (each vector = one precise fact)
  2. Cleaner input for entity extraction

Reference: Dense X Retrieval (Chen et al., EMNLP 2024)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loom.llm.provider import LLMProvider

if TYPE_CHECKING:
    from loom.ingestion.chunker import Chunk

from loom.prompts import PROPOSITION_EXTRACT


@dataclass
class Proposition:
    prop_id: str
    doc_id: str
    chunk_id: str
    text: str
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


def _extract_batch(llm: LLMProvider, batch: list[Chunk], doc_id: str) -> list[tuple[Chunk, list[str]]]:
    """Extract propositions from one batch of chunks. Returns (chunk, props) pairs."""
    combined_text = "\n\n---CHUNK BOUNDARY---\n\n".join(
        f"[Section: {c.section_title or 'untitled'}]\n{c.text}" for c in batch
    )
    prompt = PROPOSITION_EXTRACT.format(text=combined_text[:8000])
    try:
        resp = llm.generate(prompt, model="flash", temperature=0.1, max_output_tokens=4096)
        parsed = _parse_propositions(resp.text)
    except Exception:
        parsed = []

    results: list[tuple[Chunk, list[str]]] = []
    for chunk in batch:
        chunk_props = _assign_propositions_to_chunk(parsed, chunk)
        if not chunk_props:
            chunk_props = [s for s in _fallback_split(chunk.text) if len(s) >= 20]
        results.append((chunk, chunk_props))
    return results


def extract_propositions(
    llm: LLMProvider,
    chunks: list[Chunk],
    doc_id: str,
    *,
    batch_size: int = 8,
    max_workers: int = 3,
) -> list[Proposition]:
    """Extract atomic propositions from chunks using Flash with parallel batches."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

    all_batch_results: list[list[tuple[Chunk, list[str]]]] = [None] * len(batches)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_extract_batch, llm, batch, doc_id): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                all_batch_results[idx] = future.result()
            except Exception:
                all_batch_results[idx] = [
                    (chunk, [s for s in _fallback_split(chunk.text) if len(s) >= 20])
                    for chunk in batches[idx]
                ]

    all_propositions: list[Proposition] = []
    prop_idx = 0
    for batch_results in all_batch_results:
        if batch_results is None:
            continue
        for chunk, chunk_props in batch_results:
            for prop_text in chunk_props:
                prop_text = prop_text.strip()
                if len(prop_text) < 15:
                    continue
                all_propositions.append(Proposition(
                    prop_id=f"{doc_id}_p{prop_idx}",
                    doc_id=doc_id,
                    chunk_id=chunk.chunk_id,
                    text=prop_text,
                    chunk_index=chunk.chunk_index,
                ))
                prop_idx += 1

    return all_propositions


def _parse_propositions(text: str) -> list[str]:
    """Parse a JSON array of proposition strings from LLM output."""
    import json
    import re

    text = text.strip()

    for pattern in [r"```json\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1).strip()
            break

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(p) for p in parsed if isinstance(p, str) and len(p) > 10]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(p) for p in parsed if isinstance(p, str) and len(p) > 10]
        except json.JSONDecodeError:
            pass

    return []


def _assign_propositions_to_chunk(
    all_props: list[str],
    chunk,
) -> list[str]:
    """Heuristically assign propositions to the chunk they came from.

    When batching multiple chunks, propositions are interleaved.
    We assign by keyword overlap.
    """
    if not all_props:
        return []

    chunk_words = set(chunk.text.lower().split())
    scored: list[tuple[float, str]] = []
    for prop in all_props:
        prop_words = set(prop.lower().split())
        overlap = len(chunk_words & prop_words)
        scored.append((overlap, prop))

    scored.sort(reverse=True)
    threshold = max(3, len(chunk_words) * 0.05)
    return [prop for score, prop in scored if score >= threshold]


def _fallback_split(text: str) -> list[str]:
    """Split text into sentences as a fallback."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) >= 20]
