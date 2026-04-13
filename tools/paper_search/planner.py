from __future__ import annotations

import json
import logging
import time
from typing import Any

from loom.tools.paper_search.types import SearchPlan, SearchQuery
from loom.tools.paper_search.utils import extract_json_object

log = logging.getLogger(__name__)

MAX_QUERY_ANGLES = 5


def generate_search_plan(llm_provider, model: str, user_query: str) -> SearchPlan:
    """Ask the LLM to produce per-API search queries for each research angle.

    Each angle gets a tailored query string for Semantic Scholar (natural
    language), arXiv (short keyword phrases), and OpenAlex (keyword + topic).
    """
    log.info("[Planner] Generating search plan for: %s", user_query[:120])
    t0 = time.time()

    schema = {
        "queries": [
            {
                "label": "short description of this search angle",
                "semantic_scholar": "natural language query optimized for Semantic Scholar's semantic search",
                "arxiv": "short keyword phrase for arXiv (2-5 words, no boolean operators)",
                "openalex": "keyword query for OpenAlex search",
                "year_from": "integer|null",
                "year_to": "integer|null",
            }
        ],
    }
    from loom.prompts import SEARCH_PLAN
    prompt = SEARCH_PLAN.format(query=user_query, schema=json.dumps(schema, indent=2))
    response = llm_provider.generate(prompt, model=model, temperature=0.2, max_output_tokens=4096)
    elapsed = round(time.time() - t0, 2)
    log.info("[Planner] LLM responded in %.2fs", elapsed)

    parsed = extract_json_object(response.text or "")
    if not parsed:
        log.warning("[Planner] Failed to parse LLM response as JSON")
        return SearchPlan()

    queries_raw = parsed.get("queries")
    if not isinstance(queries_raw, list):
        log.warning("[Planner] LLM response missing 'queries' array")
        return SearchPlan()

    queries: list[SearchQuery] = []
    for item in queries_raw[:MAX_QUERY_ANGLES]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        s2 = str(item.get("semantic_scholar", "")).strip()
        arxiv = str(item.get("arxiv", "")).strip()
        openalex = str(item.get("openalex", "")).strip()
        if not label or not (s2 or arxiv or openalex):
            continue
        queries.append(SearchQuery(
            label=label,
            semantic_scholar=s2,
            arxiv=arxiv,
            openalex=openalex,
            year_from=item.get("year_from") if isinstance(item.get("year_from"), int) else None,
            year_to=item.get("year_to") if isinstance(item.get("year_to"), int) else None,
        ))

    for q in queries:
        log.info("[Planner]   Angle: '%s' | S2: '%s' | arXiv: '%s' | OA: '%s'",
                 q.label, q.semantic_scholar[:60], q.arxiv, q.openalex[:60])

    log.info("[Planner] Generated %d search angles in %.2fs", len(queries), elapsed)
    return SearchPlan(queries=queries)
