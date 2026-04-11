"""
Section-aware markdown chunker.

Splits documents at section boundaries (## headings) and then
sub-splits oversized sections into 500-800 token chunks, respecting
paragraph boundaries where possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


CHARS_PER_TOKEN = 4
MIN_CHUNK_TOKENS = 100
TARGET_CHUNK_TOKENS = 600
MAX_CHUNK_TOKENS = 800


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section_title: str = ""
    chunk_index: int = 0
    token_estimate: int = 0
    metadata: dict = field(default_factory=dict)


def chunk_document(
    doc_id: str,
    content: str,
    *,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> list[Chunk]:
    """Split a markdown document into section-aware chunks."""
    sections = _split_sections(content)
    chunks: list[Chunk] = []
    idx = 0

    for section_title, section_text in sections:
        section_text = section_text.strip()
        if not section_text:
            continue

        estimated = len(section_text) // CHARS_PER_TOKEN
        if estimated <= max_tokens:
            if estimated >= min_tokens:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_c{idx}",
                    doc_id=doc_id,
                    text=section_text,
                    section_title=section_title,
                    chunk_index=idx,
                    token_estimate=estimated,
                ))
                idx += 1
        else:
            sub_chunks = _split_section(section_text, target_tokens, max_tokens)
            for sc in sub_chunks:
                sc_tokens = len(sc) // CHARS_PER_TOKEN
                if sc_tokens >= min_tokens:
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}_c{idx}",
                        doc_id=doc_id,
                        text=sc,
                        section_title=section_title,
                        chunk_index=idx,
                        token_estimate=sc_tokens,
                    ))
                    idx += 1

    if not chunks and content.strip():
        text = content.strip()[:max_tokens * CHARS_PER_TOKEN]
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_c0",
            doc_id=doc_id,
            text=text,
            section_title="",
            chunk_index=0,
            token_estimate=len(text) // CHARS_PER_TOKEN,
        ))

    return chunks


def _split_sections(content: str) -> list[tuple[str, str]]:
    """Split markdown by ## headings."""
    pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(content))

    if not matches:
        return [("", content)]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = content[:matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[start:end].strip()
        sections.append((title, text))

    return sections


def _split_section(text: str, target_tokens: int, max_tokens: int) -> list[str]:
    """Sub-split a large section at paragraph boundaries."""
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    target_chars = target_tokens * CHARS_PER_TOKEN

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = len(para) // CHARS_PER_TOKEN

        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

        if current_tokens >= target_tokens:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0

    if current:
        chunks.append("\n\n".join(current))

    return chunks
