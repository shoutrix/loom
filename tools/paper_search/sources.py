"""
Paper retrieval sources: arXiv, Semantic Scholar, OpenAlex, Crossref.

Self-contained copy from parallax with all imports resolved locally.
"""

from __future__ import annotations

import html
import random
import re
import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

from loom.tools.paper_search.types import Paper
from loom.tools.paper_search.utils import normalize_text


ARXIV_API_URL = "http://export.arxiv.org/api/query"
OPENALEX_API_URL = "https://api.openalex.org/works"
CROSSREF_API_URL = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class RetryClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.last_request_time: float | None = None
        self.request_interval = 1.0
        self.max_retries = 4
        self.backoff_base = 1.0
        self.backoff_max = 30.0

    def _wait_rate_limit(self) -> None:
        now = time.time()
        if self.last_request_time is not None:
            elapsed = now - self.last_request_time
            if elapsed < self.request_interval:
                time.sleep(self.request_interval - elapsed)
        self.last_request_time = time.time()

    def _retry_wait_seconds(self, attempt: int, response: requests.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(self.backoff_max, float(retry_after))
                except ValueError:
                    pass
        raw = min(self.backoff_max, self.backoff_base * (2**attempt))
        return raw * random.uniform(0.8, 1.2)

    def get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        self._wait_rate_limit()
        for attempt in range(self.max_retries):
            response: requests.Response | None = None
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt == self.max_retries - 1:
                        response.raise_for_status()
                    time.sleep(self._retry_wait_seconds(attempt, response))
                    continue
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    return payload
                return {"data": payload}
            except requests.exceptions.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self._retry_wait_seconds(attempt, response))
        return {}

    def get_text(self, url: str, params: dict[str, Any]) -> str:
        self._wait_rate_limit()
        for attempt in range(self.max_retries):
            response: requests.Response | None = None
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt == self.max_retries - 1:
                        response.raise_for_status()
                    time.sleep(self._retry_wait_seconds(attempt, response))
                    continue
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self._retry_wait_seconds(attempt, response))
        return ""


def _paper_record(
    *,
    pid: str,
    title: str,
    abstract: str,
    source: str,
    url: str = "",
    doi: str = "",
    arxiv_id: str = "",
    year: int | None = None,
    citation_count: int = 0,
    venue: str = "",
    topic_label: str = "",
) -> Paper:
    return {
        "id": pid,
        "title": normalize_text(title),
        "abstract": normalize_text(abstract),
        "url": url,
        "doi": doi.lower().strip(),
        "arxiv_id": arxiv_id.strip(),
        "year": year,
        "citation_count": citation_count,
        "venue": venue,
        "sources": [source],
        "topic_hits": [topic_label] if topic_label else [],
        "keyword_overlap": 0.0,
        "importance_score": 0.0,
    }


class ArxivClient(RetryClient):
    def search(self, query: str, *, limit: int, topic_label: str) -> list[Paper]:
        params = {"search_query": query, "start": 0, "max_results": min(limit, 20)}
        xml_text = self.get_text(ARXIV_API_URL, params)
        if not xml_text:
            return []
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        out: list[Paper] = []
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            abstract = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            arxiv_id = url.split("/abs/")[-1] if "/abs/" in url else ""
            pid = f"arxiv:{arxiv_id or hash(title)}"
            out.append(_paper_record(
                pid=pid, title=title, abstract=abstract[:1000],
                source="arxiv", url=url, arxiv_id=arxiv_id, topic_label=topic_label,
            ))
        return out


class SemanticScholarClient(RetryClient):
    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        if api_key:
            self.session.headers.update({"x-api-key": api_key})

    def search(self, query: str, *, limit: int, topic_label: str) -> list[Paper]:
        params = {
            "query": query,
            "limit": str(min(limit, 20)),
            "fields": "paperId,title,abstract,year,url,externalIds,citationCount,venue",
        }
        data = self.get_json(SEMANTIC_SCHOLAR_API_URL, params)
        items = data.get("data", [])
        if not isinstance(items, list):
            return []
        out: list[Paper] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            ext_ids = item.get("externalIds", {}) if isinstance(item.get("externalIds"), dict) else {}
            arxiv_id = str(ext_ids.get("ArXiv", "") or "")
            doi = str(ext_ids.get("DOI", "") or "")
            pid = str(item.get("paperId", "") or "")
            if not pid:
                pid = f"s2:{hash(str(item.get('title', '')))}"
            cc = item.get("citationCount")
            out.append(_paper_record(
                pid=f"s2:{pid}",
                title=str(item.get("title", "") or ""),
                abstract=str(item.get("abstract", "") or "")[:900],
                source="semantic_scholar", url=str(item.get("url", "") or ""),
                doi=doi, arxiv_id=arxiv_id,
                year=item.get("year") if isinstance(item.get("year"), int) else None,
                citation_count=cc if isinstance(cc, int) else 0,
                venue=str(item.get("venue", "") or ""), topic_label=topic_label,
            ))
        return out

    def fetch_references(
        self, paper_id: str, limit: int = 100, *, rich: bool = False,
    ) -> list[dict[str, Any]] | list[str]:
        """Fetch papers cited BY this paper (backward references).

        With rich=True, returns dicts with paperId, isInfluential, intents.
        With rich=False (default), returns bare paperId strings for backward compat.
        """
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
        fields = "paperId,isInfluential,intents" if rich else "paperId"
        params = {"fields": fields, "limit": str(min(limit, 1000))}
        try:
            data = self.get_json(url, params)
        except Exception:
            return []
        items = data.get("data", [])
        if not isinstance(items, list):
            return []

        if rich:
            out: list[dict[str, Any]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                cited = item.get("citedPaper", {})
                if isinstance(cited, dict) and cited.get("paperId"):
                    out.append({
                        "paperId": str(cited["paperId"]),
                        "isInfluential": bool(item.get("isInfluential", False)),
                        "intents": item.get("intents", []) if isinstance(item.get("intents"), list) else [],
                    })
            return out
        else:
            out_ids: list[str] = []
            for item in items:
                if isinstance(item, dict):
                    cited = item.get("citedPaper", {})
                    if isinstance(cited, dict) and cited.get("paperId"):
                        out_ids.append(str(cited["paperId"]))
            return out_ids

    def fetch_citations(
        self, paper_id: str, limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch papers that CITE this paper (forward citations).

        Returns dicts with paperId, isInfluential, intents.
        """
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
        params = {
            "fields": "paperId,isInfluential,intents",
            "limit": str(min(limit, 1000)),
        }
        try:
            data = self.get_json(url, params)
        except Exception:
            return []
        items = data.get("data", [])
        if not isinstance(items, list):
            return []
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            citing = item.get("citingPaper", {})
            if isinstance(citing, dict) and citing.get("paperId"):
                out.append({
                    "paperId": str(citing["paperId"]),
                    "isInfluential": bool(item.get("isInfluential", False)),
                    "intents": item.get("intents", []) if isinstance(item.get("intents"), list) else [],
                })
        return out

    def fetch_recommendations(
        self,
        positive_ids: list[str],
        negative_ids: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Fetch embedding-based similar papers via the S2 Recommendations API.

        Uses BERT text embeddings + citation graph embeddings under the hood.
        """
        if len(positive_ids) == 1:
            url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{positive_ids[0]}"
            params: dict[str, Any] = {
                "limit": str(min(limit, 100)),
                "fields": "paperId,title,abstract,year,citationCount,venue,externalIds",
            }
            try:
                data = self.get_json(url, params)
            except Exception:
                return []
            items = data.get("recommendedPapers", [])
        else:
            url = "https://api.semanticscholar.org/recommendations/v1/papers/"
            body = {
                "positivePaperIds": positive_ids[:5],
            }
            if negative_ids:
                body["negativePaperIds"] = negative_ids[:5]
            try:
                self._wait_rate_limit()
                response = self.session.post(
                    url,
                    json=body,
                    params={"limit": str(min(limit, 100)),
                            "fields": "paperId,title,abstract,year,citationCount,venue,externalIds"},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
            except Exception:
                return []
            items = data.get("recommendedPapers", [])

        if not isinstance(items, list):
            return []
        out: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict) and item.get("paperId"):
                out.append(item)
        return out

    def fetch_paper_batch(
        self, paper_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Fetch full metadata for multiple papers via the batch endpoint."""
        if not paper_ids:
            return []
        url = "https://api.semanticscholar.org/graph/v1/paper/batch"
        fields = "paperId,title,abstract,year,citationCount,venue,externalIds,url"
        try:
            self._wait_rate_limit()
            response = self.session.post(
                url,
                json={"ids": paper_ids[:500]},
                params={"fields": fields},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            return []
        if isinstance(data, list):
            return [p for p in data if isinstance(p, dict) and p.get("paperId")]
        return []


def _openalex_abstract(inv_index: dict[str, list[int]]) -> str:
    parts: list[tuple[int, str]] = []
    for token, positions in inv_index.items():
        for pos in positions:
            parts.append((int(pos), token))
    parts.sort(key=lambda x: x[0])
    return " ".join(token for _, token in parts)


class OpenAlexClient(RetryClient):
    def search(self, query: str, *, limit: int, topic_label: str) -> list[Paper]:
        params = {"search": query, "per_page": min(limit, 50), "sort": "cited_by_count:desc"}
        data = self.get_json(OPENALEX_API_URL, params)
        items = data.get("results", [])
        if not isinstance(items, list):
            return []
        out: list[Paper] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            doi = str(item.get("doi", "") or "").replace("https://doi.org/", "")
            abstract = ""
            inv = item.get("abstract_inverted_index")
            if isinstance(inv, dict):
                abstract = _openalex_abstract(inv)
            pid = str(item.get("id", "") or "")
            cc = item.get("cited_by_count")
            venue = ""
            loc = item.get("primary_location")
            if isinstance(loc, dict):
                src_obj = loc.get("source")
                if isinstance(src_obj, dict):
                    venue = str(src_obj.get("display_name", "") or "")
            out.append(_paper_record(
                pid=f"openalex:{pid or hash(str(item.get('display_name', '')))}",
                title=str(item.get("display_name", "") or ""),
                abstract=abstract[:900], source="openalex",
                url=str(loc.get("landing_page_url", "") or "") if isinstance(loc, dict) else "",
                doi=doi,
                year=item.get("publication_year") if isinstance(item.get("publication_year"), int) else None,
                citation_count=cc if isinstance(cc, int) else 0,
                venue=venue, topic_label=topic_label,
            ))
        return out


class CrossrefClient(RetryClient):
    def search(self, query: str, *, limit: int, topic_label: str) -> list[Paper]:
        params = {"query": query, "rows": min(limit, 50), "sort": "relevance", "order": "desc"}
        data = self.get_json(CROSSREF_API_URL, params)
        items = data.get("message", {}).get("items", [])
        if not isinstance(items, list):
            return []
        out: list[Paper] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            titles = item.get("title", [])
            title = str(titles[0]) if isinstance(titles, list) and titles else ""
            abstract_raw = str(item.get("abstract", "") or "")
            abstract = re.sub(r"<[^>]+>", " ", html.unescape(abstract_raw))
            doi = str(item.get("DOI", "") or "")
            url = str(item.get("URL", "") or "")
            year = None
            issued = item.get("issued", {})
            if isinstance(issued, dict):
                parts = issued.get("date-parts", [])
                if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
                    y = parts[0][0]
                    if isinstance(y, int):
                        year = y
            cc = item.get("is-referenced-by-count")
            container = item.get("container-title", [])
            venue = str(container[0]) if isinstance(container, list) and container else ""
            out.append(_paper_record(
                pid=f"crossref:{doi or hash(title)}",
                title=title, abstract=abstract[:900], source="crossref",
                url=url, doi=doi, year=year,
                citation_count=cc if isinstance(cc, int) else 0,
                venue=venue, topic_label=topic_label,
            ))
        return out
