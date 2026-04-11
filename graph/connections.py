"""
Cross-domain connection discovery.

Three mechanisms:
A. Structural bridge detection (code-only, per document)
B. Embedding neighborhood latent connections (code-only, per document)
C. Pro deep connection scan (every 5 documents)

Uses novelty scoring to surface genuinely surprising connections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from loom.llm.provider import LLMProvider
    from loom.llm.embeddings import EmbeddingProvider
    from loom.graph.store import GraphStore

from loom.graph.models import BridgeEdge, LatentConnection


def detect_bridges(graph: GraphStore, new_entity_ids: list[str]) -> list[BridgeEdge]:
    """Mechanism A: Structural bridge detection.

    Any entity whose source_doc_ids span 2+ documents is a bridge.
    Zero LLM cost.
    """
    new_bridges: list[BridgeEdge] = []

    for eid in new_entity_ids:
        entity = graph.entities.get(eid)
        if entity is None:
            continue
        if len(set(entity.source_doc_ids)) >= 2:
            bridge = BridgeEdge(
                entity_id=eid,
                source_doc_ids=list(set(entity.source_doc_ids)),
            )
            graph.bridges[eid] = bridge
            new_bridges.append(bridge)

    for eid, entity in graph.entities.items():
        if eid in graph.bridges:
            continue
        if len(set(entity.source_doc_ids)) >= 2:
            bridge = BridgeEdge(
                entity_id=eid,
                source_doc_ids=list(set(entity.source_doc_ids)),
            )
            graph.bridges[eid] = bridge

    return new_bridges


def detect_latent_connections(
    graph: GraphStore,
    embedder: EmbeddingProvider,
    new_entity_ids: list[str],
    *,
    top_k: int = 5,
    similarity_threshold: float = 0.8,
) -> list[LatentConnection]:
    """Mechanism B: Embedding neighborhood discovery.

    For each new entity, find K nearest existing entities from different
    source documents. Store as weak latent connections. Zero LLM cost.
    """
    entity_ids, entity_matrix = graph.get_all_entity_embeddings()
    if len(entity_ids) < 2:
        return []

    new_connections: list[LatentConnection] = []

    for new_eid in new_entity_ids:
        new_entity = graph.entities.get(new_eid)
        if new_entity is None or new_entity.embedding is None:
            continue

        sims = entity_matrix @ new_entity.embedding
        top_indices = np.argsort(sims)[::-1][:top_k + 1]

        for idx in top_indices:
            candidate_id = entity_ids[idx]
            if candidate_id == new_eid:
                continue

            sim = float(sims[idx])
            if sim < similarity_threshold:
                break

            candidate = graph.entities.get(candidate_id)
            if candidate is None:
                continue

            new_docs = set(new_entity.source_doc_ids)
            cand_docs = set(candidate.source_doc_ids)
            if new_docs & cand_docs:
                continue

            existing = any(
                lc.entity_a_id == new_eid and lc.entity_b_id == candidate_id
                or lc.entity_a_id == candidate_id and lc.entity_b_id == new_eid
                for lc in graph.latent_connections
            )
            if existing:
                continue

            novelty = compute_novelty_score(graph, new_eid, candidate_id, sim)

            conn = LatentConnection(
                entity_a_id=new_eid,
                entity_b_id=candidate_id,
                similarity=sim,
                novelty_score=novelty,
            )
            graph.latent_connections.append(conn)
            new_connections.append(conn)

    return new_connections


def compute_novelty_score(
    graph: GraphStore,
    entity_a_id: str,
    entity_b_id: str,
    embedding_similarity: float,
) -> float:
    """Novelty = embedding_similarity * graph_distance_penalty * community_diversity.

    High novelty = semantically similar but structurally distant and in different domains.
    """
    path_len = graph.shortest_path_length(entity_a_id, entity_b_id, max_depth=10)
    if path_len < 0:
        graph_distance_penalty = 1.0
    else:
        graph_distance_penalty = min(path_len / 5.0, 1.0)

    entity_a = graph.entities.get(entity_a_id)
    entity_b = graph.entities.get(entity_b_id)
    if entity_a and entity_b:
        if entity_a.community_id != entity_b.community_id and entity_a.community_id >= 0:
            community_diversity = 1.0
        else:
            community_diversity = 0.3
    else:
        community_diversity = 0.5

    return embedding_similarity * graph_distance_penalty * community_diversity


def deep_connection_scan(
    llm: LLMProvider,
    graph: GraphStore,
    *,
    max_candidates: int = 10,
) -> list[LatentConnection]:
    """Mechanism C: Pro deep connection analysis.

    Analyzes accumulated bridge entities and latent connection candidates.
    Pro reasons about WHY connections exist.
    """
    unconfirmed = [
        lc for lc in graph.latent_connections
        if not lc.confirmed and lc.novelty_score > 0.3
    ]
    unconfirmed.sort(key=lambda x: x.novelty_score, reverse=True)
    candidates = unconfirmed[:max_candidates]

    if not candidates:
        return []

    pairs_text: list[str] = []
    for lc in candidates:
        entity_a = graph.entities.get(lc.entity_a_id)
        entity_b = graph.entities.get(lc.entity_b_id)
        if not entity_a or not entity_b:
            continue

        pairs_text.append(
            f"- Entity A: \"{entity_a.name}\" ({entity_a.entity_type}): {entity_a.description}\n"
            f"  Sources: {', '.join(entity_a.source_doc_ids[:3])}\n"
            f"  Entity B: \"{entity_b.name}\" ({entity_b.entity_type}): {entity_b.description}\n"
            f"  Sources: {', '.join(entity_b.source_doc_ids[:3])}\n"
            f"  Embedding similarity: {lc.similarity:.2f}"
        )

    if not pairs_text:
        return []

    prompt = (
        "You are analyzing potential cross-domain connections in a research knowledge graph.\n\n"
        "For each pair of entities from different research domains, determine:\n"
        "1. Is there a meaningful intellectual connection? (not just superficial keyword overlap)\n"
        "2. If yes, explain the deep connection in 2-3 sentences\n"
        "3. Could understanding from one domain inform the other?\n\n"
        "Pairs:\n" + "\n\n".join(pairs_text) + "\n\n"
        "Return a JSON array with one entry per pair:\n"
        '[{"connected": true, "description": "..."}, {"connected": false, "description": ""}, ...]'
    )

    resp = llm.generate(prompt, model="pro", temperature=0.3, max_output_tokens=4096)
    confirmed: list[LatentConnection] = []

    try:
        import json, re
        text = resp.text.strip()
        for pat in [r"```json\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
            m = re.search(pat, text, re.DOTALL)
            if m:
                text = m.group(1).strip()
                break
        parsed = json.loads(text)
        if isinstance(parsed, list):
            for i, decision in enumerate(parsed):
                if i >= len(candidates):
                    break
                lc = candidates[i]
                if isinstance(decision, dict) and decision.get("connected"):
                    lc.confirmed = True
                    lc.description = str(decision.get("description", ""))

                    graph.add_relationship(
                        source_id=lc.entity_a_id,
                        target_id=lc.entity_b_id,
                        relation_type="deep_connection",
                        description=lc.description,
                        weight=lc.novelty_score,
                    )
                    confirmed.append(lc)
    except Exception as e:
        print(f"  [Connections] Deep scan parse failed: {e}")

    return confirmed
