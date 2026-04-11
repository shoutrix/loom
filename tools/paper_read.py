"""
Paper reader: fetch full text of a paper and ingest into the knowledge graph.

Supports arXiv papers, DOIs, Semantic Scholar IDs, and generic URLs.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from loom.ingestion.pipeline import IngestionPipeline, IngestionResult


@dataclass
class PaperReadResult:
    doc_id: str
    title: str
    source_type: str
    ingestion_result: IngestionResult | None = None
    error: str = ""


def read_and_ingest_paper(
    pipeline: IngestionPipeline,
    identifier: str,
) -> PaperReadResult:
    """Read a paper by arXiv ID, DOI, S2 ID, or URL and ingest into the knowledge graph."""
    from loom.ingestion.parsers import parse_arxiv, parse_url, ParsedDocument

    identifier = identifier.strip()

    try:
        if _is_arxiv_identifier(identifier):
            parsed = parse_arxiv(identifier)
        elif _is_doi(identifier):
            parsed = _parse_doi(identifier)
        elif identifier.startswith("s2:"):
            parsed = _parse_s2_id(identifier[3:])
        elif identifier.startswith("http"):
            parsed = parse_url(identifier)
        else:
            parsed = _parse_doi(identifier) if "/" in identifier else parse_url(identifier)

        result = pipeline.ingest_document(parsed)
        return PaperReadResult(
            doc_id=parsed.doc_id,
            title=parsed.title,
            source_type=parsed.source_type,
            ingestion_result=result,
        )

    except Exception as e:
        return PaperReadResult(doc_id="", title="", source_type="error", error=str(e))


def _is_arxiv_identifier(s: str) -> bool:
    s = s.strip()
    if "arxiv.org" in s:
        return True
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", s):
        return True
    return False


def _is_doi(s: str) -> bool:
    return bool(re.match(r"^10\.\d{4,}/", s))


def _parse_doi(doi: str) -> Any:
    """Resolve a DOI: try full text via doi.org redirect, fall back to S2 metadata."""
    from loom.ingestion.parsers import parse_url, ParsedDocument

    doi = doi.strip()
    url = f"https://doi.org/{doi}"

    try:
        parsed = parse_url(url)
        if len(parsed.content) > 500:
            parsed.doc_id = f"doi_{hashlib.sha256(doi.encode()).hexdigest()[:12]}"
            return parsed
    except Exception:
        pass

    return _fetch_s2_metadata(f"DOI:{doi}", doi_str=doi)


def _parse_s2_id(s2_id: str) -> Any:
    """Fetch paper metadata from Semantic Scholar and create a ParsedDocument."""
    return _fetch_s2_metadata(s2_id)


def _fetch_s2_metadata(paper_ref: str, doi_str: str = "") -> Any:
    """Fetch title + abstract from Semantic Scholar API and build a ParsedDocument."""
    from loom.ingestion.parsers import ParsedDocument

    api_key = os.getenv("CORTEX_SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"x-api-key": api_key} if api_key else {}

    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_ref}"
    params = {"fields": "title,abstract,year,venue,externalIds,url"}

    resp = requests.get(url, params=params, headers=headers, timeout=15)
    if resp.status_code != 200:
        raise ValueError(f"S2 API returned {resp.status_code} for {paper_ref}")

    data = resp.json()
    title = data.get("title", "") or ""
    abstract = data.get("abstract", "") or ""
    year = data.get("year", "")
    venue = data.get("venue", "")
    s2_url = data.get("url", "")

    if not title and not abstract:
        raise ValueError(f"No metadata found for {paper_ref}")

    content_parts = [f"# {title}"]
    if year or venue:
        content_parts.append(f"\n*{venue} {year}*\n")
    if abstract:
        content_parts.append(f"## Abstract\n\n{abstract}")

    content = "\n".join(content_parts)

    ext_ids = data.get("externalIds", {}) or {}
    arxiv_id = ext_ids.get("ArXiv", "")
    s2_id = data.get("paperId", paper_ref)

    doc_id = f"s2_{hashlib.sha256(s2_id.encode()).hexdigest()[:12]}"
    if arxiv_id:
        doc_id = f"arxiv_{arxiv_id.replace('/', '_').replace('.', '_')}"

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        content=content,
        source_type="s2_metadata",
        source_url=s2_url or f"https://doi.org/{doi_str}" if doi_str else s2_url,
        abstract=abstract,
    )
