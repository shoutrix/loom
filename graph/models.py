"""
Core data models for the knowledge graph.

Entity, Relationship, Community, BridgeEdge, LatentConnection.
"""

from __future__ import annotations

import base64
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Entity:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    entity_type: str = "concept"
    description: str = ""
    mentions: int = 1
    source_doc_ids: list[str] = field(default_factory=list)
    community_id: int = -1
    embedding: np.ndarray | None = field(default=None, repr=False)
    embedding_model: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def confidence(self) -> float:
        source_div = min(len(set(self.source_doc_ids)), 3)
        if source_div == 0:
            source_div_score = 0.3
        elif source_div == 1:
            source_div_score = 0.5
        elif source_div == 2:
            source_div_score = 0.8
        else:
            source_div_score = 1.0

        type_specificity = 1.0
        if self.entity_type in ("concept",):
            type_specificity = 0.5

        return math.log2(1 + self.mentions) * source_div_score * type_specificity

    @property
    def name_entropy(self) -> float:
        """Shannon entropy of the entity name (used for entropy guard)."""
        if not self.name:
            return 0.0
        freq: dict[str, int] = {}
        for c in self.name.lower():
            freq[c] = freq.get(c, 0) + 1
        total = len(self.name.lower())
        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def to_dict(self, include_embedding: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type,
            "description": self.description,
            "mentions": self.mentions,
            "source_doc_ids": self.source_doc_ids,
            "community_id": self.community_id,
            "confidence": round(self.confidence, 3),
            "embedding_model": self.embedding_model,
        }
        if include_embedding and self.embedding is not None:
            d["embedding_b64"] = base64.b64encode(self.embedding.astype(np.float32).tobytes()).decode("ascii")
            d["embedding_dim"] = int(self.embedding.shape[0])
        return d

    @staticmethod
    def embedding_from_b64(b64: str, dim: int) -> np.ndarray:
        raw = base64.b64decode(b64)
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(dim)


@dataclass
class Relationship:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_id: str = ""
    target_id: str = ""
    relation_type: str = "related_to"
    description: str = ""
    weight: float = 1.0
    source_doc_id: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.relation_type,
            "description": self.description,
            "weight": self.weight,
            "source_doc_id": self.source_doc_id,
        }


@dataclass
class Community:
    id: int
    entity_ids: list[str] = field(default_factory=list)
    summary: str = ""
    key_concepts: list[str] = field(default_factory=list)
    dirty: bool = True
    entity_count_at_last_summary: int = 0
    last_updated: str = ""

    @property
    def needs_resummarization(self) -> bool:
        if self.dirty:
            return True
        if (
            self.entity_count_at_last_summary > 0
            and len(self.entity_ids) > 2 * self.entity_count_at_last_summary
        ):
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity_ids": self.entity_ids,
            "size": len(self.entity_ids),
            "key_concepts": self.key_concepts,
            "summary": self.summary,
            "dirty": self.dirty,
            "entity_count_at_last_summary": self.entity_count_at_last_summary,
        }


@dataclass
class BridgeEdge:
    entity_id: str
    source_doc_ids: list[str] = field(default_factory=list)
    novelty_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "source_doc_ids": self.source_doc_ids,
            "novelty_score": round(self.novelty_score, 3),
        }


@dataclass
class LatentConnection:
    entity_a_id: str
    entity_b_id: str
    similarity: float
    confirmed: bool = False
    description: str = ""
    novelty_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_a_id": self.entity_a_id,
            "entity_b_id": self.entity_b_id,
            "similarity": round(self.similarity, 3),
            "confirmed": self.confirmed,
            "description": self.description,
            "novelty_score": round(self.novelty_score, 3),
        }


@dataclass
class MergeLogEntry:
    merged_from_id: str
    merged_into_id: str
    merged_from_name: str
    merged_into_name: str
    reason: str
    timestamp: str
