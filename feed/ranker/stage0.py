"""
Stage 0 ranker -- cold-start heuristic (used until ~15 ratings).

  score = 0.5  * seed_topic_sim
        + 0.20 * source_quality
        + 0.15 * llm_relevance
        + 0.10 * novelty
        + 0.05 * recency
        + 0.10 * (engagement_prior or citation_prior)   # crowd signal bonus

Each feature is in [0, 1]. `llm_relevance` is 0 unless LLM scoring ran.
The crowd-signal bonus uses whichever applies to the item kind (capped at 0.10
effective weight, in line with the plan's source-quality cap principle).
"""

from __future__ import annotations

from loom.feed.features import FeatureVector


def score(fv: FeatureVector) -> float:
    crowd = max(fv.engagement_prior, fv.citation_prior)
    return (
        0.50 * fv.seed_topic_sim
        + 0.20 * fv.source_quality
        + 0.15 * fv.llm_relevance
        + 0.10 * fv.novelty
        + 0.05 * fv.recency
        + 0.10 * crowd
    )
