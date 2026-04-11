from __future__ import annotations

import json
from typing import Any

from loom.tools.paper_search.types import SearchPlan, SearchTopic
from loom.tools.paper_search.utils import extract_json_object

MAX_TOPICS = 4


def generate_search_plan(llm_provider, model: str, query: str) -> SearchPlan:
    schema = {
        "topics": [
            {"label": "string", "keywords": ["string"], "year_from": "integer|null", "year_to": "integer|null"}
        ],
    }
    prompt = (
        "You are an expert research planner for academic paper discovery.\n"
        "Break the query into 3-4 distinct research topics with strong search keywords.\n"
        "Keywords must be concrete and academic, optimized for arXiv/Semantic Scholar.\n\n"
        f"User Query: {query}\n\n"
        f"JSON schema:\n{json.dumps(schema, indent=2)}"
    )
    response = llm_provider.generate(prompt, model=model, temperature=0.1, max_output_tokens=2048)
    parsed = extract_json_object(response.text or "")
    if not parsed:
        return SearchPlan()

    topics_raw = parsed.get("topics")
    if not isinstance(topics_raw, list):
        return SearchPlan()

    topics: list[SearchTopic] = []
    for topic in topics_raw[:MAX_TOPICS]:
        if not isinstance(topic, dict):
            continue
        label = str(topic.get("label", "")).strip()
        kws = topic.get("keywords", [])
        if not label or not isinstance(kws, list):
            continue
        keywords = [str(k).strip() for k in kws if str(k).strip()][:12]
        if not keywords:
            continue
        topics.append(SearchTopic(
            label=label, keywords=keywords,
            year_from=topic.get("year_from") if isinstance(topic.get("year_from"), int) else None,
            year_to=topic.get("year_to") if isinstance(topic.get("year_to"), int) else None,
        ))

    return SearchPlan(topics=topics, global_constraints=parsed.get("global_constraints", {}))
