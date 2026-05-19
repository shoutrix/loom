"""
Feature computation for feed candidates.

Each feature is a float in [0, 1]. They are assembled into a per-item
feature vector that the ranker fuses into a final score.

Phase 3 features:
  - seed_topic_sim     -- cosine to seed-topic embedding
  - sim_pos            -- cosine to positives centroid (0 if no positives)
  - sim_neg            -- cosine to negatives centroid (0 if no negatives)
  - personalization_delta = sim_pos - sim_neg
  - source_quality     -- per-source prior, capped
  - recency            -- exponential decay on age
  - novelty            -- 1 - max cos to items shown in last 14 days
  - engagement_prior   -- log(HN points + 1) normalized (blogs only)
  - citation_prior     -- log(citation count + 1) normalized (papers only)
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from loom.feed import db
from loom.feed.models import Candidate, FeedProfile


# ---- source quality priors -------------------------------------------------

# Allowlist for high-quality blog domains. Anything else gets the default.
_BLOG_DOMAIN_QUALITY: dict[str, float] = {
    "distill.pub": 1.0,
    "lilianweng.github.io": 1.0,
    "sebastianraschka.com": 0.95,
    "magazine.sebastianraschka.com": 0.95,
    "jalammar.github.io": 0.95,
    "gwern.net": 0.95,
    "huggingface.co": 0.85,
    "openai.com": 0.85,
    "anthropic.com": 0.85,
    "deepmind.google": 0.85,
    "research.google": 0.85,
    "arxiv-sanity-lite.com": 0.7,
    "stratechery.com": 0.85,
    "matt-rickard.com": 0.7,
    "interconnects.ai": 0.85,
    "thegradient.pub": 0.85,
    "ruder.io": 0.9,
    "karpathy.github.io": 1.0,
    "fast.ai": 0.85,
    "blog.research.google": 0.85,
    "bair.berkeley.edu": 0.9,
    "ai.googleblog.com": 0.85,
    "machinelearningmastery.com": 0.55,
}

_PAPER_SOURCE_QUALITY: dict[str, float] = {
    "arxiv": 0.6,
    "semantic_scholar": 0.7,
    "openalex": 0.6,
    "serper_scholar": 0.6,
}


def source_quality(c: Candidate) -> float:
    if c.kind == "paper":
        return _PAPER_SOURCE_QUALITY.get(c.source, 0.5)
    # blog
    if c.source == "rss":
        # take feed_url's domain
        feed_url = c.extra.get("feed_url") or c.url
        domain = _domain_of(feed_url)
        return _BLOG_DOMAIN_QUALITY.get(domain, 0.45)
    if c.source == "hn":
        domain = _domain_of(c.url)
        return _BLOG_DOMAIN_QUALITY.get(domain, 0.45)
    return 0.4


def _domain_of(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).hostname.lower().lstrip("www.") if urlparse(url).hostname else ""
    except Exception:
        return ""


# ---- recency ---------------------------------------------------------------

def recency(published_at: str, *, half_life_days: float = 30.0) -> float:
    """Exponential decay on item age. half_life_days controls steepness."""
    if not published_at:
        return 0.5
    try:
        ts = datetime.datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.UTC)
    except Exception:
        return 0.5
    age_days = max(0.0, (datetime.datetime.now(datetime.UTC) - ts).total_seconds() / 86400.0)
    return float(0.5 ** (age_days / half_life_days))  # halves every `half_life_days`


# ---- engagement / citation priors ------------------------------------------

def engagement_prior(c: Candidate) -> float:
    if c.kind != "blog":
        return 0.0
    points = float(c.extra.get("points") or 0)
    comments = float(c.extra.get("num_comments") or 0)
    raw = math.log(1.0 + points + 0.5 * comments)
    return min(1.0, raw / 6.0)  # ~ log(400) cap


def citation_prior(c: Candidate) -> float:
    if c.kind != "paper":
        return 0.0
    citations = float(c.extra.get("citation_count") or 0)
    return min(1.0, math.log(1.0 + citations) / 8.0)  # ~ log(3000) cap


# ---- embedding-based features ----------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    val = float(np.dot(a, b) / (na * nb))
    # cosine in [-1,1]; remap to [0,1] for use as a feature
    return max(0.0, (val + 1.0) / 2.0)


def seed_topic_sim(item_embedding: np.ndarray, seed_embedding: np.ndarray | None) -> float:
    if seed_embedding is None:
        return 0.5
    return cosine(item_embedding, seed_embedding)


def personalization(
    item_embedding: np.ndarray,
    pos_centroid: np.ndarray | None,
    neg_centroid: np.ndarray | None,
) -> tuple[float, float, float]:
    """Returns (sim_pos, sim_neg, personalization_delta in [0,1])."""
    sim_p = cosine(item_embedding, pos_centroid) if pos_centroid is not None else 0.0
    sim_n = cosine(item_embedding, neg_centroid) if neg_centroid is not None else 0.0
    delta_raw = sim_p - sim_n  # in [-1, 1]
    delta = max(0.0, (delta_raw + 1.0) / 2.0)
    return sim_p, sim_n, delta


def novelty(
    item_embedding: np.ndarray,
    recent_embeddings: list[np.ndarray],
) -> float:
    if not recent_embeddings:
        return 1.0
    max_sim = 0.0
    for e in recent_embeddings:
        s = cosine(item_embedding, e)
        if s > max_sim:
            max_sim = s
    return max(0.0, 1.0 - max_sim)


# ---- assembly --------------------------------------------------------------

@dataclass
class FeatureVector:
    seed_topic_sim: float = 0.0
    sim_pos: float = 0.0
    sim_neg: float = 0.0
    personalization_delta: float = 0.0
    source_quality: float = 0.0
    recency: float = 0.0
    novelty: float = 0.0
    engagement_prior: float = 0.0
    citation_prior: float = 0.0
    llm_relevance: float = 0.0  # filled later if LLM scoring runs

    def as_dict(self) -> dict[str, float]:
        return {
            "seed_topic_sim": self.seed_topic_sim,
            "sim_pos": self.sim_pos,
            "sim_neg": self.sim_neg,
            "personalization_delta": self.personalization_delta,
            "source_quality": self.source_quality,
            "recency": self.recency,
            "novelty": self.novelty,
            "engagement_prior": self.engagement_prior,
            "citation_prior": self.citation_prior,
            "llm_relevance": self.llm_relevance,
        }


def compute_features(
    candidate: Candidate,
    item_embedding: np.ndarray,
    *,
    seed_embedding: np.ndarray | None,
    pos_centroid: np.ndarray | None,
    neg_centroid: np.ndarray | None,
    recent_embeddings: list[np.ndarray],
) -> FeatureVector:
    sim_p, sim_n, delta = personalization(item_embedding, pos_centroid, neg_centroid)
    return FeatureVector(
        seed_topic_sim=seed_topic_sim(item_embedding, seed_embedding),
        sim_pos=sim_p,
        sim_neg=sim_n,
        personalization_delta=delta,
        source_quality=source_quality(candidate),
        recency=recency(candidate.published_at),
        novelty=novelty(item_embedding, recent_embeddings),
        engagement_prior=engagement_prior(candidate),
        citation_prior=citation_prior(candidate),
    )
