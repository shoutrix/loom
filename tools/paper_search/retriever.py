from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from loom.tools.paper_search.sources import (
    ArxivClient,
    CrossrefClient,
    OpenAlexClient,
    SemanticScholarClient,
)
from loom.tools.paper_search.types import Paper, SearchPlan
from loom.tools.paper_search.defaults import MAX_KEYWORDS_PER_TOPIC


def _source_query(topic_keywords: list[str], topic_label: str) -> str:
    joined = " ".join(topic_keywords[:MAX_KEYWORDS_PER_TOPIC]).strip()
    if not joined:
        return topic_label
    return f"{topic_label} {joined}".strip()


class RetrievalDispatcher:
    def __init__(self, *, semantic_scholar_api_key: str | None = None, max_workers: int = 8) -> None:
        self.clients = {
            "arxiv": ArxivClient(),
            "semantic_scholar": SemanticScholarClient(api_key=semantic_scholar_api_key),
            "openalex": OpenAlexClient(),
            "crossref": CrossrefClient(),
        }
        self.max_workers = max_workers

    def retrieve(self, plan: SearchPlan, per_source_limit: int) -> list[Paper]:
        tasks: list[tuple[str, str, str, int]] = []
        for topic in plan.topics:
            query = _source_query(topic.keywords, topic.label)
            for source in self.clients:
                tasks.append((source, query, topic.label, per_source_limit))

        papers: list[Paper] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_map = {
                ex.submit(self.clients[source].search, query, limit=limit, topic_label=topic): source
                for source, query, topic, limit in tasks
            }
            for fut in as_completed(future_map):
                try:
                    res = fut.result()
                except Exception:
                    continue
                if isinstance(res, list):
                    papers.extend([p for p in res if isinstance(p, dict)])
        return papers


def merge_topic_keywords(plan: SearchPlan) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for topic in plan.topics:
        out[topic.label] = topic.keywords
    return out


def plan_from_gap_queries(gap_queries: list[str]) -> SearchPlan:
    from loom.tools.paper_search.types import SearchTopic
    topics = [SearchTopic(label=q[:80], keywords=q.split()[:10]) for q in gap_queries if q.strip()]
    return SearchPlan(topics=topics)
