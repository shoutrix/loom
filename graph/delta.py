"""
Delta filter: only ingest genuinely new information.

Checks if extracted entities/relationships already exist in the graph
before creating duplicates. Also handles semantic caching for
near-duplicate chunks.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from loom.graph.store import GraphStore


def filter_new_entities(
    graph: GraphStore,
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove entities that already exist exactly in the graph."""
    new_entities: list[dict[str, Any]] = []
    for ent in entities:
        name = ent.get("name", "").strip().lower()
        existing = graph.get_entity_by_name(name)
        if existing is None:
            new_entities.append(ent)
    return new_entities


def filter_duplicate_relationships(
    graph: GraphStore,
    relationships: list[dict[str, Any]],
    uuid_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Remove relationships that already exist in the graph."""
    existing_triples: set[tuple[str, str, str]] = set()
    for rel in graph.relationships:
        existing_triples.add((rel.source_id, rel.target_id, rel.relation_type))

    new_rels: list[dict[str, Any]] = []
    for rel in relationships:
        src = uuid_map.get(rel.get("source_name", "").lower())
        tgt = uuid_map.get(rel.get("target_name", "").lower())
        rtype = rel.get("type", "related_to")
        if src and tgt and (src, tgt, rtype) not in existing_triples:
            new_rels.append(rel)

    return new_rels


def is_near_duplicate_chunk(
    chunk_embedding: np.ndarray,
    existing_embeddings: np.ndarray,
    threshold: float = 0.95,
) -> bool:
    """Check if a chunk is a near-duplicate of existing chunks."""
    if existing_embeddings.shape[0] == 0:
        return False
    sims = existing_embeddings @ chunk_embedding
    return bool(np.max(sims) >= threshold)
