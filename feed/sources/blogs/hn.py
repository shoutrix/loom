"""Hacker News source adapter via Algolia search API."""

from __future__ import annotations

import datetime
import logging
from typing import Any

import requests

from loom.feed.models import Candidate

log = logging.getLogger(__name__)


_ENDPOINT = "https://hn.algolia.com/api/v1/search_by_date"
_DEFAULT_TIMEOUT = 20.0


def fetch(
    query: str,
    *,
    since: datetime.datetime | None = None,
    until: datetime.datetime | None = None,
    limit: int = 50,
    min_points: int = 5,
) -> list[Candidate]:
    """Search HN stories by date matching `query`."""
    numeric_filters = []
    if since:
        numeric_filters.append(f"created_at_i>{int(since.timestamp())}")
    if until:
        numeric_filters.append(f"created_at_i<{int(until.timestamp())}")
    if min_points:
        numeric_filters.append(f"points>={min_points}")

    params: dict[str, Any] = {
        "tags": "story",
        "hitsPerPage": limit,
    }
    if query:
        params["query"] = query
    if numeric_filters:
        params["numericFilters"] = ",".join(numeric_filters)

    try:
        r = requests.get(_ENDPOINT, params=params, timeout=_DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("[hn] fetch failed: %s", e)
        return []

    out: list[Candidate] = []
    for hit in data.get("hits", []):
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        title = (hit.get("title") or "").strip()
        if not title:
            continue
        published = ""
        if hit.get("created_at"):
            published = hit["created_at"]
        out.append(Candidate(
            kind="blog",
            source="hn",
            external_id=str(hit.get("objectID") or ""),
            url=url,
            title=title,
            authors=[hit.get("author")] if hit.get("author") else [],
            published_at=published,
            abstract=(hit.get("story_text") or "").strip()[:1500],
            extra={
                "points": int(hit.get("points") or 0),
                "num_comments": int(hit.get("num_comments") or 0),
                "hn_id": hit.get("objectID"),
            },
        ))
    return out
