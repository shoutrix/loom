from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from loom.tools.paper_search.sources import (
    ArxivClient,
    OpenAlexClient,
    SemanticScholarClient,
)
from loom.tools.paper_search.types import Paper, SearchPlan
from loom.tools.paper_search.defaults import (
    S2_RESULTS_PER_QUERY,
    ARXIV_RESULTS_PER_QUERY,
    OPENALEX_RESULTS_PER_QUERY,
)

log = logging.getLogger(__name__)


class RetrievalDispatcher:
    def __init__(self, *, semantic_scholar_api_key: str | None = None, max_workers: int = 12) -> None:
        self.s2 = SemanticScholarClient(api_key=semantic_scholar_api_key)
        self.arxiv = ArxivClient()
        self.openalex = OpenAlexClient()
        self.max_workers = max_workers

    def retrieve(self, plan: SearchPlan, raw_query: str) -> list[Paper]:
        """Dispatch per-API queries from a SearchPlan + a raw-query pass."""
        tasks: list[tuple[str, str, str, int]] = []

        for sq in plan.queries:
            if sq.semantic_scholar:
                tasks.append(("s2", sq.semantic_scholar, sq.label, S2_RESULTS_PER_QUERY))
            if sq.arxiv:
                tasks.append(("arxiv", sq.arxiv, sq.label, ARXIV_RESULTS_PER_QUERY))
            if sq.openalex:
                tasks.append(("openalex", sq.openalex, sq.label, OPENALEX_RESULTS_PER_QUERY))

        tasks.append(("s2", raw_query, "direct_query", S2_RESULTS_PER_QUERY))
        tasks.append(("openalex", raw_query, "direct_query", OPENALEX_RESULTS_PER_QUERY))

        return self._dispatch(tasks)

    def retrieve_from_angles(self, angles: list[dict[str, Any]], raw_query: str) -> list[Paper]:
        """Dispatch per-API queries from pre-built angle dicts (discovery read output).

        Each angle dict has keys: label, semantic_scholar, arxiv, openalex.
        """
        tasks: list[tuple[str, str, str, int]] = []

        for angle in angles:
            label = str(angle.get("label", ""))
            s2 = str(angle.get("semantic_scholar", "")).strip()
            arxiv = str(angle.get("arxiv", "")).strip()
            openalex = str(angle.get("openalex", "")).strip()
            if s2:
                tasks.append(("s2", s2, label, S2_RESULTS_PER_QUERY))
            if arxiv:
                tasks.append(("arxiv", arxiv, label, ARXIV_RESULTS_PER_QUERY))
            if openalex:
                tasks.append(("openalex", openalex, label, OPENALEX_RESULTS_PER_QUERY))

        tasks.append(("s2", raw_query, "direct_query", S2_RESULTS_PER_QUERY))
        tasks.append(("openalex", raw_query, "direct_query", OPENALEX_RESULTS_PER_QUERY))

        return self._dispatch(tasks)

    def _dispatch(self, tasks: list[tuple[str, str, str, int]]) -> list[Paper]:
        papers: list[Paper] = []

        def _run_search(source: str, query: str, angle: str, limit: int) -> list[Paper]:
            if source == "s2":
                return self.s2.search(query, limit=limit, search_angle=angle)
            elif source == "arxiv":
                return self.arxiv.search(query, limit=limit, search_angle=angle)
            elif source == "openalex":
                return self.openalex.search(query, limit=limit, search_angle=angle)
            return []

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_map = {
                ex.submit(_run_search, src, q, angle, lim): (src, angle)
                for src, q, angle, lim in tasks
            }
            for fut in as_completed(future_map):
                src, angle = future_map[fut]
                try:
                    res = fut.result()
                    if isinstance(res, list):
                        papers.extend([p for p in res if isinstance(p, dict)])
                        log.info("[Retriever] %s/%s returned %d papers", src, angle, len(res))
                except Exception as e:
                    log.warning("[Retriever] %s/%s failed: %s", src, angle, e)

        log.info("[Retriever] Total raw papers: %d from %d tasks", len(papers), len(tasks))
        return papers
