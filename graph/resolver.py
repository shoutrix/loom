"""
Three-tier entity resolution.

Tier 1: Exact name match (code-only, ~70%)
Tier 2: Embedding + fuzzy + Flash gray-zone (~15%)
Tier 3: Pro batch for genuinely ambiguous cases (~10%)

Includes entropy guard: low-entropy names skip Tier 2 -> Tier 3 directly.
"""

from __future__ import annotations

import json
import re
import datetime
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from loom.config import GraphSettings
    from loom.llm.provider import LLMProvider
    from loom.llm.embeddings import EmbeddingProvider
    from loom.graph.store import GraphStore

from loom.graph.models import Entity


def resolve_entities(
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    graph: GraphStore,
    raw_entities: list[dict[str, Any]],
    entity_embeddings: np.ndarray | None,
    doc_id: str,
    config: GraphSettings,
) -> tuple[dict[str, str], list[str]]:
    """Resolve extracted entities against the existing graph.

    Returns:
        uuid_map: dict mapping entity name (lowercase) -> resolved entity ID
        new_entity_ids: list of IDs for newly created entities
    """
    uuid_map: dict[str, str] = {}
    new_entity_ids: list[str] = []
    tier3_batch: list[dict[str, Any]] = []
    tier2_gray: list[dict[str, Any]] = []

    for i, raw_ent in enumerate(raw_entities):
        name = raw_ent["name"]
        normalized = name.strip().lower()
        embedding = entity_embeddings[i] if entity_embeddings is not None and i < len(entity_embeddings) else None

        # Tier 1: Exact match
        existing = graph.get_entity_by_name(name)
        if existing is not None:
            new_ent = _build_entity(raw_ent, doc_id, embedding, embedder)
            graph.merge_entity(existing.id, new_ent, reason="exact_match")
            uuid_map[normalized] = existing.id
            continue

        # Entropy guard: low-entropy names skip to Tier 3
        temp_ent = Entity(name=name)
        if temp_ent.name_entropy < config.entropy_min_threshold or len(name) < config.entropy_min_name_length:
            tier3_batch.append({**raw_ent, "_index": i, "_embedding": embedding})
            continue

        # Tier 2: Embedding + fuzzy
        if embedding is not None and len(graph.entities) > 0:
            candidate = _find_embedding_candidate(
                graph, embedding, name, raw_ent.get("type", "concept"),
                config.entity_embedding_similarity_threshold,
                config.fuzzy_match_threshold,
            )
            if candidate is not None:
                if candidate["confidence"] == "high":
                    new_ent = _build_entity(raw_ent, doc_id, embedding, embedder)
                    graph.merge_entity(candidate["entity_id"], new_ent, reason="embedding_fuzzy_match")
                    uuid_map[normalized] = candidate["entity_id"]
                    continue
                else:
                    tier2_gray.append({
                        **raw_ent,
                        "_index": i,
                        "_embedding": embedding,
                        "_candidate": candidate,
                    })
                    continue

        # No candidates -- create new entity
        new_ent = _build_entity(raw_ent, doc_id, embedding, embedder)
        graph.add_entity(new_ent)
        uuid_map[normalized] = new_ent.id
        new_entity_ids.append(new_ent.id)

    # Tier 2 gray-zone: Flash confirmation
    if tier2_gray:
        flash_results = _flash_gray_zone_resolve(llm, tier2_gray)
        for item, decision in zip(tier2_gray, flash_results):
            normalized = item["name"].strip().lower()
            embedding = item["_embedding"]
            if decision == "merge":
                candidate_id = item["_candidate"]["entity_id"]
                new_ent = _build_entity(item, doc_id, embedding, embedder)
                graph.merge_entity(candidate_id, new_ent, reason="flash_gray_zone_merge")
                uuid_map[normalized] = candidate_id
            else:
                new_ent = _build_entity(item, doc_id, embedding, embedder)
                graph.add_entity(new_ent)
                uuid_map[normalized] = new_ent.id
                new_entity_ids.append(new_ent.id)

    # Tier 3: LLM batch resolution (processed in small batches)
    if tier3_batch:
        # If graph is nearly empty, skip LLM -- nothing to merge against
        if len(graph.entities) < 5:
            for item in tier3_batch:
                normalized = item["name"].strip().lower()
                embedding = item["_embedding"]
                new_ent = _build_entity(item, doc_id, embedding, embedder)
                graph.add_entity(new_ent)
                uuid_map[normalized] = new_ent.id
                new_entity_ids.append(new_ent.id)
        else:
            batch_size = 15
            all_results: list[str | None] = []
            for start in range(0, len(tier3_batch), batch_size):
                chunk = tier3_batch[start : start + batch_size]
                chunk_results = _pro_batch_resolve(llm, embedder, graph, chunk, doc_id)
                all_results.extend(chunk_results)

            for item, resolved_id in zip(tier3_batch, all_results):
                normalized = item["name"].strip().lower()
                embedding = item["_embedding"]
                if resolved_id:
                    new_ent = _build_entity(item, doc_id, embedding, embedder)
                    graph.merge_entity(resolved_id, new_ent, reason="pro_resolution")
                    uuid_map[normalized] = resolved_id
                else:
                    new_ent = _build_entity(item, doc_id, embedding, embedder)
                    graph.add_entity(new_ent)
                    uuid_map[normalized] = new_ent.id
                    new_entity_ids.append(new_ent.id)

    return uuid_map, new_entity_ids


def _build_entity(
    raw: dict[str, Any],
    doc_id: str,
    embedding: np.ndarray | None,
    embedder: EmbeddingProvider | None,
) -> Entity:
    return Entity(
        name=raw["name"],
        entity_type=raw.get("type", "concept"),
        description=raw.get("description", ""),
        source_doc_ids=[doc_id],
        embedding=embedding,
        embedding_model=embedder.model if embedder else "",
        created_at=datetime.datetime.now().isoformat(),
        updated_at=datetime.datetime.now().isoformat(),
    )


def _find_embedding_candidate(
    graph: GraphStore,
    embedding: np.ndarray,
    name: str,
    entity_type: str,
    sim_threshold: float,
    fuzzy_threshold: float,
) -> dict[str, Any] | None:
    """Search for embedding-similar entities with fuzzy name check."""
    entity_ids, entity_matrix = graph.get_all_entity_embeddings()
    if len(entity_ids) == 0:
        return None

    similarities = entity_matrix @ embedding
    top_indices = np.argsort(similarities)[::-1][:5]

    for idx in top_indices:
        sim = float(similarities[idx])
        if sim < sim_threshold * 0.9:
            break

        candidate_id = entity_ids[idx]
        candidate = graph.entities[candidate_id]

        if candidate.entity_type != entity_type and entity_type != "concept":
            continue

        fuzzy_score = _jaccard_ngrams(name.lower(), candidate.name.lower(), n=3)

        if sim >= sim_threshold and fuzzy_score >= fuzzy_threshold:
            return {
                "entity_id": candidate_id,
                "name": candidate.name,
                "similarity": sim,
                "fuzzy_score": fuzzy_score,
                "confidence": "high",
            }
        elif sim >= sim_threshold * 0.95:
            return {
                "entity_id": candidate_id,
                "name": candidate.name,
                "similarity": sim,
                "fuzzy_score": fuzzy_score,
                "confidence": "gray",
            }

    return None


def _jaccard_ngrams(a: str, b: str, n: int = 3) -> float:
    if not a or not b:
        return 0.0
    ngrams_a = set(a[i:i+n] for i in range(len(a) - n + 1))
    ngrams_b = set(b[i:i+n] for i in range(len(b) - n + 1))
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b
    return len(intersection) / len(union)


def _flash_gray_zone_resolve(
    llm: LLMProvider,
    items: list[dict[str, Any]],
) -> list[str]:
    """Batch Flash call for gray-zone entity merge decisions."""
    if not items:
        return []

    pairs_text = []
    for item in items:
        candidate = item["_candidate"]
        pairs_text.append(
            f"- New: \"{item['name']}\" ({item.get('type', 'concept')}): {item.get('description', '')}\n"
            f"  Candidate: \"{candidate['name']}\" (similarity={candidate['similarity']:.2f})"
        )

    prompt = (
        "For each pair, decide if the new entity should MERGE with the candidate "
        "or be kept as a SEPARATE entity.\n\n"
        "Pairs:\n" + "\n".join(pairs_text) + "\n\n"
        "Reply with a JSON array of decisions, one per pair: [\"merge\", \"separate\", ...]"
    )

    resp = llm.generate(prompt, model="flash", temperature=0.1, max_output_tokens=512)
    try:
        parsed = json.loads(resp.text.strip().strip("```json").strip("```").strip())
        if isinstance(parsed, list):
            decisions = [str(d).lower() for d in parsed]
            while len(decisions) < len(items):
                decisions.append("separate")
            return decisions
    except (json.JSONDecodeError, ValueError):
        pass

    return ["separate"] * len(items)


def _pro_batch_resolve(
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    graph: GraphStore,
    items: list[dict[str, Any]],
    doc_id: str,
) -> list[str | None]:
    """Batch Pro call for ambiguous entity resolution."""
    if not items:
        return []

    entity_ids, entity_matrix = graph.get_all_entity_embeddings()
    entries_text: list[str] = []

    for item in items:
        embedding = item.get("_embedding")
        candidates_text = "No existing candidates."

        if embedding is not None and len(entity_ids) > 0:
            sims = entity_matrix @ embedding
            top_indices = np.argsort(sims)[::-1][:3]
            candidates = []
            for idx in top_indices:
                if sims[idx] > 0.5:
                    eid = entity_ids[idx]
                    ent = graph.entities[eid]
                    candidates.append(
                        f"  [{eid}] \"{ent.name}\" ({ent.entity_type}): {ent.description} "
                        f"(similarity={sims[idx]:.2f})"
                    )
            if candidates:
                candidates_text = "\n".join(candidates)

        entries_text.append(
            f"Entity: \"{item['name']}\" ({item.get('type', 'concept')}): {item.get('description', '')}\n"
            f"Candidates:\n{candidates_text}"
        )

    prompt = (
        "You are resolving entity references in a knowledge graph.\n"
        "For each entity, decide:\n"
        "- MERGE with candidate [id] if they refer to the same concept\n"
        "- NEW if this is a genuinely distinct concept\n\n"
        "Consider context carefully. Same name ≠ same concept "
        "(e.g., 'transformer' in ML vs electrical engineering).\n\n"
        + "\n\n".join(f"--- Entity {i+1} ---\n{t}" for i, t in enumerate(entries_text))
        + "\n\nReturn a JSON array with one entry per entity: "
        "[{\"action\": \"merge\", \"target_id\": \"xxx\"}, {\"action\": \"new\"}, ...]"
    )

    # Use flash for resolution -- pro's thinking budget consumes too many tokens
    resp = llm.generate(prompt, model="flash", temperature=0.1, max_output_tokens=2048)

    try:
        text = resp.text.strip()
        for pat in [r"```json\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
            m = re.search(pat, text, re.DOTALL)
            if m:
                text = m.group(1).strip()
                break
        parsed = json.loads(text)
        if isinstance(parsed, list):
            results: list[str | None] = []
            for decision in parsed:
                if isinstance(decision, dict) and decision.get("action") == "merge":
                    target_id = decision.get("target_id", "")
                    if target_id in graph.entities:
                        results.append(target_id)
                    else:
                        results.append(None)
                else:
                    results.append(None)
            while len(results) < len(items):
                results.append(None)
            return results
    except (json.JSONDecodeError, ValueError):
        pass

    return [None] * len(items)
