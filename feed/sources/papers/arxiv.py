"""arXiv source adapter with date filter."""

from __future__ import annotations

import datetime
import logging
import time
import urllib.parse
from typing import Any

import requests

from loom.feed.models import Candidate

log = logging.getLogger(__name__)


_ARXIV_API = "http://export.arxiv.org/api/query"
_DEFAULT_TIMEOUT = 30.0
_LAST_CALL = 0.0
_MIN_INTERVAL_S = 3.0  # arxiv rate limit


def _rate_limit() -> None:
    global _LAST_CALL
    elapsed = time.time() - _LAST_CALL
    if elapsed < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - elapsed)
    _LAST_CALL = time.time()


def fetch(
    query: str,
    *,
    since: datetime.datetime | None = None,
    until: datetime.datetime | None = None,
    limit: int = 50,
) -> list[Candidate]:
    """Query arXiv API for papers matching `query` in optional [since, until]."""
    _rate_limit()

    parts = [f"all:{query}"] if query else []
    if since or until:
        s = (since or datetime.datetime(1990, 1, 1)).strftime("%Y%m%d%H%M")
        u = (until or datetime.datetime.now()).strftime("%Y%m%d%H%M")
        parts.append(f"submittedDate:[{s} TO {u}]")
    search_query = "+AND+".join(parts) if parts else f"all:{query}"

    params = {
        "search_query": search_query,
        "max_results": limit,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{_ARXIV_API}?{urllib.parse.urlencode(params)}"

    try:
        r = requests.get(url, timeout=_DEFAULT_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        log.warning("[arxiv] fetch failed: %s", e)
        return []

    return _parse_atom(r.text)


def _parse_atom(xml_text: str) -> list[Candidate]:
    import xml.etree.ElementTree as ET

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    out: list[Candidate] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        log.warning("[arxiv] parse error: %s", e)
        return out

    for entry in root.findall("atom:entry", ns):
        arxiv_id_full = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        # http://arxiv.org/abs/2401.12345v1 -> 2401.12345
        arxiv_id = arxiv_id_full.rsplit("/", 1)[-1].split("v")[0] if arxiv_id_full else ""
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip().replace("\n", " ")
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
        authors = [
            (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
            for a in entry.findall("atom:author", ns)
        ]
        url = next(
            (
                link.attrib.get("href", "")
                for link in entry.findall("atom:link", ns)
                if link.attrib.get("type") == "application/pdf"
                or link.attrib.get("rel") == "alternate"
            ),
            arxiv_id_full,
        )

        if not arxiv_id or not title:
            continue

        out.append(Candidate(
            kind="paper",
            source="arxiv",
            external_id=arxiv_id,
            url=url or arxiv_id_full,
            title=title,
            authors=[a for a in authors if a],
            published_at=published,
            abstract=summary,
            extra={"arxiv_id": arxiv_id},
        ))
    return out
