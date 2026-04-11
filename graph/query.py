"""
Graph RAG: context assembly for Q&A.

Retrieves relevant entity context, community summaries, and
cross-domain connections for grounding LLM answers.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loom.graph.store import GraphStore


def retrieve_graph_context(
    graph: GraphStore,
    entity_names: list[str] | None = None,
    query_keywords: list[str] | None = None,
    max_communities: int = 5,
    max_entities: int = 15,
) -> str:
    """Assemble graph context for a Q&A answer."""
    lines: list[str] = []

    # Entity context
    entity_ids: set[str] = set()
    if entity_names:
        for name in entity_names:
            entity = graph.get_entity_by_name(name)
            if entity:
                entity_ids.add(entity.id)

    if entity_ids:
        expanded: set[str] = set(entity_ids)
        for eid in entity_ids:
            for neighbor_id in graph.get_entity_neighbors(eid)[:5]:
                expanded.add(neighbor_id)

        lines.append("=== RELEVANT ENTITIES ===")
        for eid in list(expanded)[:max_entities]:
            entity = graph.entities.get(eid)
            if not entity:
                continue
            prefix = " *" if eid in entity_ids else "  "
            lines.append(f"{prefix} {entity.name} ({entity.entity_type}): {entity.description}")

            rels = graph.get_entity_relationships(eid)[:5]
            for rel in rels:
                other_id = rel.target_id if rel.source_id == eid else rel.source_id
                other = graph.entities.get(other_id)
                if other:
                    if rel.source_id == eid:
                        lines.append(f"     --[{rel.relation_type}]--> {other.name}")
                    else:
                        lines.append(f"     <--[{rel.relation_type}]-- {other.name}")

    # Community summaries
    if graph.communities:
        target_communities: set[int] = set()
        for eid in entity_ids:
            e = graph.entities.get(eid)
            if e and e.community_id >= 0:
                target_communities.add(e.community_id)

        scored: list[tuple[float, int]] = []
        for cid, community in graph.communities.items():
            if not community.summary:
                continue
            score = float(len(community.entity_ids))
            if cid in target_communities:
                score += 20.0
            if query_keywords:
                text = community.summary.lower() + " " + " ".join(community.key_concepts).lower()
                score += sum(3.0 for kw in query_keywords if kw.lower() in text)
            scored.append((score, cid))

        scored.sort(reverse=True)
        top = scored[:max_communities]

        if top:
            lines.append("\n=== KNOWLEDGE CLUSTERS ===")
            for _score, cid in top:
                community = graph.communities[cid]
                concepts = ", ".join(community.key_concepts[:5])
                lines.append(f"\n[Cluster: {concepts}]")
                lines.append(community.summary)

    # Bridge connections
    bridges = [b for b in graph.bridges.values() if b.entity_id in entity_ids]
    if bridges:
        lines.append("\n=== CROSS-DOCUMENT BRIDGES ===")
        for bridge in bridges[:5]:
            entity = graph.entities.get(bridge.entity_id)
            if entity:
                docs = ", ".join(bridge.source_doc_ids[:3])
                lines.append(f"  {entity.name}: appears in documents [{docs}]")

    confirmed_connections = [
        lc for lc in graph.latent_connections
        if lc.confirmed and (lc.entity_a_id in entity_ids or lc.entity_b_id in entity_ids)
    ]
    if confirmed_connections:
        lines.append("\n=== DEEP CROSS-DOMAIN CONNECTIONS ===")
        for lc in confirmed_connections[:5]:
            ea = graph.entities.get(lc.entity_a_id)
            eb = graph.entities.get(lc.entity_b_id)
            if ea and eb:
                lines.append(f"  {ea.name} <--> {eb.name}: {lc.description}")

    return "\n".join(lines) if lines else ""


def find_entities_in_text(graph: GraphStore, text: str) -> list[str]:
    """Find known entity names that appear in the given text."""
    text_lower = text.lower()
    found: list[str] = []
    for name_lower, eid in graph._name_index.items():
        if len(name_lower) < 3:
            continue
        if name_lower in text_lower:
            entity = graph.entities.get(eid)
            if entity and entity.confidence >= 0.5:
                found.append(entity.name)
    return found
