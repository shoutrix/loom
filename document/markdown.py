"""Pure helpers over raw markdown bodies.

Used by `submit_document` (fast title fallback) and by the metadata
worker (arxiv-id extraction for research_paper references).
"""

from __future__ import annotations

import re


_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n?", re.DOTALL)
_H1_RE = re.compile(r"^\s{0,3}#\s+(.+?)\s*$", re.MULTILINE)
_ARXIV_RE = re.compile(
    r"(?:arxiv[:\s]*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)


def strip_frontmatter(body: str) -> str:
    """Remove a leading YAML frontmatter block, if present."""
    return _FRONTMATTER_RE.sub("", body or "", count=1)


def extract_title_from_markdown(body: str) -> str:
    """Best-effort title: first H1 inside the body (after frontmatter).

    Returns empty string when no H1 is found. The metadata worker takes
    over from there with an LLM call.
    """
    stripped = strip_frontmatter(body or "")
    m = _H1_RE.search(stripped)
    if not m:
        return ""
    return m.group(1).strip()[:300]


def extract_arxiv_references(body: str) -> list[str]:
    """Pull every arxiv id mentioned in the body.

    Returns the de-duplicated, order-preserving list of bare ids
    (e.g. "1706.03762"). The metadata worker passes these to the
    citation-graph builder to populate Reference.doc_id.
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in _ARXIV_RE.finditer(body or ""):
        bare = m.group(1)
        if bare not in seen:
            seen.add(bare)
            out.append(bare)
    return out
