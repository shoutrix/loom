"""RSS / Atom feed source adapter. Pulls recent entries from configured feeds."""

from __future__ import annotations

import datetime
import hashlib
import logging
from typing import Any

import feedparser

from loom.recommender.models import Candidate

log = logging.getLogger(__name__)


def fetch(
    feed_urls: list[str],
    *,
    since: datetime.datetime | None = None,
    until: datetime.datetime | None = None,
    per_feed_limit: int = 20,
) -> list[Candidate]:
    """Fetch entries from a list of RSS/Atom feed URLs."""
    out: list[Candidate] = []
    for feed_url in feed_urls:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception as e:
            log.warning("[rss] fetch failed for %s: %s", feed_url, e)
            continue

        feed_title = (parsed.feed.get("title") if parsed.feed else "") or feed_url

        for entry in parsed.entries[:per_feed_limit]:
            published = _parse_entry_date(entry)
            if since and published and published < since:
                continue
            if until and published and published > until:
                continue

            url = entry.get("link", "")
            title = (entry.get("title") or "").strip()
            if not url or not title:
                continue

            external_id = entry.get("id") or hashlib.md5(url.encode()).hexdigest()
            authors = []
            if entry.get("author"):
                authors.append(entry.get("author"))
            elif entry.get("authors"):
                authors = [a.get("name") for a in entry.get("authors") if a.get("name")]

            abstract = ""
            if entry.get("summary"):
                abstract = _strip_html(entry["summary"])[:1500]
            elif entry.get("content"):
                content = entry["content"]
                if isinstance(content, list) and content:
                    abstract = _strip_html(content[0].get("value") or "")[:1500]

            out.append(Candidate(
                kind="blog",
                source="rss",
                external_id=str(external_id),
                url=url,
                title=title,
                authors=[a for a in authors if a],
                published_at=published.isoformat() if published else "",
                abstract=abstract,
                extra={"feed": feed_title, "feed_url": feed_url},
            ))
    return out


def _parse_entry_date(entry: dict[str, Any]) -> datetime.datetime | None:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        v = entry.get(key)
        if v:
            try:
                return datetime.datetime(*v[:6], tzinfo=datetime.UTC)
            except Exception:
                continue
    return None


def _strip_html(text: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", text or "").strip()
