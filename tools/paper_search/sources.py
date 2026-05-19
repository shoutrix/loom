"""
Paper retrieval sources: arXiv, Semantic Scholar, OpenAlex, Google Scholar (via Serper).
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from typing import Any

import json as _json
import requests

from loom.tools.paper_search.types import Paper
from loom.tools.paper_search.utils import normalize_text

log = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
OPENALEX_API_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class RetryClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.last_request_time_by_host: dict[str, float] = {}
        self.interval_by_host: dict[str, float] = {
            "api.semanticscholar.org": 1.05,
            "export.arxiv.org": 3.0,
            "api.openalex.org": 0.5,
        }
        self.default_request_interval = 0.4
        self.max_retries = 4
        self.backoff_base = 1.0
        self.backoff_max = 30.0
        self._rate_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._throttled_429: dict[str, dict[str, float]] = {}

    def _wait_rate_limit(self, url: str) -> None:
        host = urlparse(url).netloc.lower()
        interval = self.interval_by_host.get(host, self.default_request_interval)
        while True:
            with self._rate_lock:
                now = time.time()
                last = self.last_request_time_by_host.get(host)
                if last is None:
                    self.last_request_time_by_host[host] = now
                    return
                elapsed = now - last
                wait_for = interval - elapsed
                if wait_for <= 0:
                    self.last_request_time_by_host[host] = now
                    return
            time.sleep(wait_for)

    def _log_retry(self, url: str, status_code: int, attempt: int, wait: float) -> None:
        base = url.split("?")[0]
        if status_code != 429:
            log.warning("[HTTP] %s returned %d, retry %d/%d in %.1fs",
                        base, status_code, attempt + 1, self.max_retries, wait)
            return

        # Collapse repetitive 429 lines per endpoint into periodic summaries.
        now = time.time()
        with self._log_lock:
            state = self._throttled_429.setdefault(base, {"last": 0.0, "suppressed": 0.0})
            if now - state["last"] >= 10.0:
                suppressed = int(state["suppressed"])
                if suppressed > 0:
                    log.warning("[HTTP] %s returned 429, retry %d/%d in %.1fs (%d similar 429s suppressed)",
                                base, attempt + 1, self.max_retries, wait, suppressed)
                else:
                    log.warning("[HTTP] %s returned 429, retry %d/%d in %.1fs",
                                base, attempt + 1, self.max_retries, wait)
                state["last"] = now
                state["suppressed"] = 0.0
            else:
                state["suppressed"] += 1

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
        self._wait_rate_limit(url)
        for attempt in range(self.max_retries):
            response: requests.Response | None = None
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code in (429, 500, 502, 503, 504):
                    wait = self._retry_wait_seconds(attempt, response)
                    self._log_retry(url, response.status_code, attempt, wait)
                    if attempt == self.max_retries - 1:
                        response.raise_for_status()
                    time.sleep(wait)
                    continue
                if 400 <= response.status_code < 500:
                    # Non-retriable client errors (except 429) should fail fast.
                    response.raise_for_status()
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    return payload
                return {"data": payload}
            except requests.exceptions.RequestException as e:
                if response is not None and 400 <= response.status_code < 500 and response.status_code != 429:
                    log.warning("[HTTP] %s non-retriable client error %d: %s",
                                url.split("?")[0], response.status_code, e)
                    raise
                if attempt == self.max_retries - 1:
                    log.error("[HTTP] %s failed after %d retries: %s", url.split("?")[0], self.max_retries, e)
                    raise
                wait = self._retry_wait_seconds(attempt, response)
                log.warning("[HTTP] %s error: %s, retry %d/%d in %.1fs",
                            url.split("?")[0], e, attempt + 1, self.max_retries, wait)
                time.sleep(wait)
        return {}

    def get_text(self, url: str, params: dict[str, Any]) -> str:
        self._wait_rate_limit(url)
        for attempt in range(self.max_retries):
            response: requests.Response | None = None
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code in (429, 500, 502, 503, 504):
                    wait = self._retry_wait_seconds(attempt, response)
                    self._log_retry(url, response.status_code, attempt, wait)
                    if attempt == self.max_retries - 1:
                        response.raise_for_status()
                    time.sleep(wait)
                    continue
                if 400 <= response.status_code < 500:
                    # Non-retriable client errors (except 429) should fail fast.
                    response.raise_for_status()
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                if response is not None and 400 <= response.status_code < 500 and response.status_code != 429:
                    log.warning("[HTTP] %s non-retriable client error %d: %s",
                                url.split("?")[0], response.status_code, e)
                    raise
                if attempt == self.max_retries - 1:
                    log.error("[HTTP] %s failed after %d retries: %s", url.split("?")[0], self.max_retries, e)
                    raise
                wait = self._retry_wait_seconds(attempt, response)
                log.warning("[HTTP] %s error: %s, retry %d/%d in %.1fs",
                            url.split("?")[0], e, attempt + 1, self.max_retries, wait)
                time.sleep(wait)
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
    search_angle: str = "",
    authors: list[dict[str, str]] | None = None,
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
        "search_angles": [search_angle] if search_angle else [],
        "importance_score": 0.0,
        "authors": authors or [],
    }


class ArxivClient(RetryClient):
    def search(self, query: str, *, limit: int = 15, search_angle: str = "") -> list[Paper]:
        log.info("[arXiv] Searching: '%s' (limit=%d, angle='%s')", query[:80], limit, search_angle)
        t0 = time.time()
        params = {"search_query": query, "start": 0, "max_results": min(limit, 30)}
        xml_text = self.get_text(ARXIV_API_URL, params)
        if not xml_text:
            log.warning("[arXiv] Empty response for query: '%s'", query[:80])
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
                source="arxiv", url=url, arxiv_id=arxiv_id, search_angle=search_angle,
            ))
        log.info("[arXiv] Found %d papers in %.2fs for '%s'", len(out), time.time() - t0, query[:60])
        return out


class SemanticScholarClient(RetryClient):
    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        if api_key:
            self.session.headers.update({"x-api-key": api_key})
            log.info("[S2] Initialized with API key")
        else:
            log.info("[S2] Initialized without API key (rate limits will be stricter)")

    def search(self, query: str, *, limit: int = 50, search_angle: str = "") -> list[Paper]:
        log.info("[S2] Searching: '%s' (limit=%d, angle='%s')", query[:80], limit, search_angle)
        t0 = time.time()
        params = {
            "query": query,
            "limit": str(min(limit, 100)),
            "fields": "paperId,title,abstract,year,url,externalIds,citationCount,venue,authors",
        }
        data = self.get_json(SEMANTIC_SCHOLAR_API_URL, params)
        items = data.get("data", [])
        if not isinstance(items, list):
            log.warning("[S2] Unexpected response format for query: '%s'", query[:80])
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
            raw_authors = item.get("authors", [])
            authors = []
            if isinstance(raw_authors, list):
                for a in raw_authors:
                    if isinstance(a, dict) and a.get("authorId"):
                        authors.append({
                            "authorId": str(a["authorId"]),
                            "name": str(a.get("name", "")),
                        })
            out.append(_paper_record(
                pid=f"s2:{pid}",
                title=str(item.get("title", "") or ""),
                abstract=str(item.get("abstract", "") or "")[:900],
                source="semantic_scholar", url=str(item.get("url", "") or ""),
                doi=doi, arxiv_id=arxiv_id,
                year=item.get("year") if isinstance(item.get("year"), int) else None,
                citation_count=cc if isinstance(cc, int) else 0,
                venue=str(item.get("venue", "") or ""), search_angle=search_angle,
                authors=authors,
            ))
        log.info("[S2] Found %d papers in %.2fs for '%s'", len(out), time.time() - t0, query[:60])
        return out

    def fetch_references(
        self, paper_id: str, limit: int = 100, *, rich: bool = False,
    ) -> list[dict[str, Any]] | list[str]:
        log.debug("[S2] Fetching references for %s (rich=%s)", paper_id[:20], rich)
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
        fields = "paperId,isInfluential,intents" if rich else "paperId"
        params = {"fields": fields, "limit": str(min(limit, 1000))}
        try:
            data = self.get_json(url, params)
        except Exception as e:
            log.warning("[S2] Failed to fetch references for %s: %s", paper_id[:20], e)
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
            influential_count = sum(1 for r in out if r.get("isInfluential"))
            log.debug("[S2] References for %s: %d total, %d influential", paper_id[:20], len(out), influential_count)
            return out
        else:
            out_ids: list[str] = []
            for item in items:
                if isinstance(item, dict):
                    cited = item.get("citedPaper", {})
                    if isinstance(cited, dict) and cited.get("paperId"):
                        out_ids.append(str(cited["paperId"]))
            log.debug("[S2] References for %s: %d IDs", paper_id[:20], len(out_ids))
            return out_ids

    def fetch_citations(
        self, paper_id: str, limit: int = 100,
    ) -> list[dict[str, Any]]:
        log.debug("[S2] Fetching citations for %s", paper_id[:20])
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
        params = {
            "fields": "paperId,isInfluential,intents",
            "limit": str(min(limit, 1000)),
        }
        try:
            data = self.get_json(url, params)
        except Exception as e:
            log.warning("[S2] Failed to fetch citations for %s: %s", paper_id[:20], e)
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
        influential_count = sum(1 for c in out if c.get("isInfluential"))
        log.debug("[S2] Citations for %s: %d total, %d influential", paper_id[:20], len(out), influential_count)
        return out

    def fetch_recommendations(
        self,
        positive_ids: list[str],
        negative_ids: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        log.info("[S2] Fetching recommendations (positives=%d, negatives=%d, limit=%d)",
                 len(positive_ids), len(negative_ids or []), limit)
        t0 = time.time()
        if len(positive_ids) == 1:
            url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{positive_ids[0]}"
            params: dict[str, Any] = {
                "limit": str(min(limit, 100)),
                "fields": "paperId,title,abstract,year,citationCount,venue,externalIds",
            }
            try:
                data = self.get_json(url, params)
            except Exception as e:
                log.warning("[S2] Recommendations failed: %s", e)
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
                self._wait_rate_limit(url)
                response = self.session.post(
                    url,
                    json=body,
                    params={"limit": str(min(limit, 100)),
                            "fields": "paperId,title,abstract,year,citationCount,venue,externalIds"},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                log.warning("[S2] Recommendations failed: %s", e)
                return []
            items = data.get("recommendedPapers", [])

        if not isinstance(items, list):
            return []
        out: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict) and item.get("paperId"):
                out.append(item)
        log.info("[S2] Got %d recommendations in %.2fs", len(out), time.time() - t0)
        return out

    def fetch_paper_batch(
        self, paper_ids: list[str],
    ) -> list[dict[str, Any]]:
        if not paper_ids:
            return []
        log.info("[S2] Batch-fetching metadata for %d papers", len(paper_ids))
        t0 = time.time()
        url = "https://api.semanticscholar.org/graph/v1/paper/batch"
        fields = "paperId,title,abstract,year,citationCount,venue,externalIds,url,authors"
        try:
            self._wait_rate_limit(url)
            response = self.session.post(
                url,
                json={"ids": paper_ids[:500]},
                params={"fields": fields},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            log.warning("[S2] Batch fetch failed: %s", e)
            return []
        if isinstance(data, list):
            result = [p for p in data if isinstance(p, dict) and p.get("paperId")]
            log.info("[S2] Batch returned %d/%d papers in %.2fs", len(result), len(paper_ids), time.time() - t0)
            return result
        log.warning("[S2] Batch returned unexpected type: %s", type(data).__name__)
        return []


def _openalex_abstract(inv_index: dict[str, list[int]]) -> str:
    parts: list[tuple[int, str]] = []
    for token, positions in inv_index.items():
        for pos in positions:
            parts.append((int(pos), token))
    parts.sort(key=lambda x: x[0])
    return " ".join(token for _, token in parts)


_ARXIV_DOI_RE = re.compile(r"10\.48550/arXiv\.(\d{4}\.\d{4,5})", re.IGNORECASE)


def _openalex_extract_arxiv_id(item: dict, doi: str, landing_url: str) -> str:
    """Try to extract an arXiv ID from an OpenAlex work via DOI, locations, or IDs."""
    # 1. arXiv DOI (e.g. 10.48550/arXiv.2409.12117)
    m = _ARXIV_DOI_RE.search(doi)
    if m:
        return m.group(1)
    # 2. Landing page URL (e.g. https://arxiv.org/abs/2409.12117)
    if "arxiv.org" in landing_url:
        parts = landing_url.split("/abs/")
        if len(parts) == 2:
            return parts[1].split("v")[0].strip()
    # 3. Check all locations for arXiv URLs
    for loc in (item.get("locations") or []):
        if not isinstance(loc, dict):
            continue
        for key in ("landing_page_url", "pdf_url"):
            url = str(loc.get(key, "") or "")
            if "arxiv.org" in url:
                parts = url.split("/abs/")
                if len(parts) == 2:
                    return parts[1].split("v")[0].strip()
                parts = url.split("/pdf/")
                if len(parts) == 2:
                    return parts[1].split("v")[0].strip().rstrip(".pdf")
    return ""


class OpenAlexClient(RetryClient):
    def search(self, query: str, *, limit: int = 30, search_angle: str = "") -> list[Paper]:
        log.info("[OpenAlex] Searching: '%s' (limit=%d, angle='%s')", query[:80], limit, search_angle)
        t0 = time.time()
        params = {"search": query, "per_page": min(limit, 50), "sort": "cited_by_count:desc"}
        data = self.get_json(OPENALEX_API_URL, params)
        items = data.get("results", [])
        if not isinstance(items, list):
            log.warning("[OpenAlex] Unexpected response format for query: '%s'", query[:80])
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
            landing_url = ""
            if isinstance(loc, dict):
                src_obj = loc.get("source")
                if isinstance(src_obj, dict):
                    venue = str(src_obj.get("display_name", "") or "")
                landing_url = str(loc.get("landing_page_url", "") or "")

            arxiv_id = _openalex_extract_arxiv_id(item, doi, landing_url)

            out.append(_paper_record(
                pid=f"openalex:{pid or hash(str(item.get('display_name', '')))}",
                title=str(item.get("display_name", "") or ""),
                abstract=abstract[:900], source="openalex",
                url=landing_url,
                doi=doi, arxiv_id=arxiv_id,
                year=item.get("publication_year") if isinstance(item.get("publication_year"), int) else None,
                citation_count=cc if isinstance(cc, int) else 0,
                venue=venue, search_angle=search_angle,
            ))
        log.info("[OpenAlex] Found %d papers in %.2fs for '%s'", len(out), time.time() - t0, query[:60])
        return out


# ── Google Scholar via Serper ─────────────────────────────────────────

SERPER_SCHOLAR_URL = "https://google.serper.dev/scholar"
SERPER_WEB_URL = "https://google.serper.dev/search"

_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")
_DOI_URL_RE = re.compile(r"doi\.org/(10\.\d{4,9}/[^\s]+)")


def _extract_arxiv_id_from_url(url: str) -> str:
    m = _ARXIV_URL_RE.search(url)
    return m.group(1) if m else ""


def _extract_doi_from_url(url: str) -> str:
    m = _DOI_URL_RE.search(url)
    return m.group(1) if m else ""


def _parse_year_from_summary(summary: str) -> int | None:
    """Try to pull a 4-digit year from the publication_info summary."""
    m = re.search(r"\b(19|20)\d{2}\b", summary)
    if m:
        return int(m.group(0))
    return None


class SerperScholarClient:
    """Google Scholar search via the Serper /scholar endpoint.

    Only instantiated when SERPER_API_KEY is available.
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()
        log.info("[Serper] Initialized Google Scholar client")

    def search(self, query: str, *, num: int = 20, search_angle: str = "") -> list[Paper]:
        log.info("[Serper] Scholar search: '%s' (num=%d, angle='%s')", query[:80], num, search_angle)
        t0 = time.time()
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": min(num, 40)}
        try:
            resp = self.session.post(SERPER_SCHOLAR_URL, headers=headers,
                                     data=_json.dumps(payload), timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning("[Serper] Scholar search failed: %s", e)
            return []

        organic = data.get("organic", [])
        if not isinstance(organic, list):
            log.warning("[Serper] Unexpected response format")
            return []

        out: list[Paper] = []
        for idx, item in enumerate(organic):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue

            link = str(item.get("link", ""))
            snippet = str(item.get("snippet", ""))
            arxiv_id = _extract_arxiv_id_from_url(link)
            doi = _extract_doi_from_url(link)

            # Also check pdfUrl for arXiv links
            pdf_url = str(item.get("pdfUrl", ""))
            if not arxiv_id and pdf_url:
                arxiv_id = _extract_arxiv_id_from_url(pdf_url)

            # Serper uses camelCase "publicationInfo"
            pub_info = item.get("publicationInfo", "") or item.get("publication_info", "")
            pub_str = str(pub_info) if pub_info else ""

            authors: list[dict[str, str]] = []
            if pub_str:
                # Format: "A Author, B Author - Journal, Year - Publisher"
                author_part = pub_str.split(" - ")[0] if " - " in pub_str else ""
                for name in author_part.split(", "):
                    name = name.strip()
                    if name and len(name) > 1:
                        authors.append({"authorId": "", "name": name})

            # Top-level citedBy and year
            cited_by_val = item.get("citedBy", 0)
            citation_count = int(cited_by_val) if isinstance(cited_by_val, (int, float)) else 0

            year_val = item.get("year")
            year: int | None = int(year_val) if isinstance(year_val, (int, float)) else None
            if year is not None and not (1900 <= year <= 2100):
                year = None
            if year is None:
                year = _parse_year_from_summary(pub_str)

            pid = f"serper:{arxiv_id or doi or hash(title)}"
            out.append(_paper_record(
                pid=pid,
                title=title,
                abstract=snippet[:900],
                source="google_scholar",
                url=link,
                doi=doi,
                arxiv_id=arxiv_id,
                year=year,
                citation_count=citation_count,
                venue="",
                search_angle=search_angle,
                authors=authors,
            ))

        log.info("[Serper] Found %d papers in %.2fs for '%s'", len(out), time.time() - t0, query[:60])
        return out

    def web_search(self, query: str, *, num: int = 10) -> list[dict[str, str]]:
        """Regular Google web search. Returns lightweight dicts (title + snippet + link) for vocabulary discovery."""
        log.info("[Serper] Web search: '%s' (num=%d)", query[:80], num)
        t0 = time.time()
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": min(num, 20)}
        try:
            resp = self.session.post(SERPER_WEB_URL, headers=headers,
                                     data=_json.dumps(payload), timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning("[Serper] Web search failed: %s", e)
            return []

        out: list[dict[str, str]] = []
        for item in data.get("organic", []):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            link = str(item.get("link", "")).strip()
            if title and snippet:
                out.append({"title": title, "snippet": snippet, "link": link})

        log.info("[Serper] Web search returned %d results in %.2fs", len(out), time.time() - t0)
        return out
