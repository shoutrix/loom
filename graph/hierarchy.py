"""
Incremental community detection with lazy re-summarization.

Uses union-find for connected components. Only re-summarizes
communities that are dirty (had entities added/removed).
Uses Pro for community summaries (deep synthesis task).
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loom.llm.provider import LLMProvider
    from loom.graph.store import GraphStore

from loom.graph.models import Community


def update_communities(
    llm: LLMProvider,
    graph: GraphStore,
) -> int:
    """Run incremental community detection and re-summarize dirty communities.

    Returns the number of communities that were re-summarized.
    """
    _detect_communities_incremental(graph)
    return _resummarize_dirty(llm, graph)


def _detect_communities_incremental(graph: GraphStore) -> None:
    """Recompute communities using union-find."""
    if not graph.entities:
        return

    parent: dict[str, str] = {eid: eid for eid in graph.entities}
    rank: dict[str, int] = {eid: 0 for eid in graph.entities}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for rel in graph.relationships:
        if rel.source_id in parent and rel.target_id in parent:
            union(rel.source_id, rel.target_id)

    groups: dict[str, list[str]] = defaultdict(list)
    for eid in graph.entities:
        root = find(eid)
        groups[root].append(eid)

    old_communities = dict(graph.communities)
    old_entity_community: dict[str, int] = {}
    for cid, comm in old_communities.items():
        for eid in comm.entity_ids:
            old_entity_community[eid] = cid

    graph.communities.clear()
    graph._next_community_id = 0

    for _root, member_ids in groups.items():
        old_cids = set()
        for eid in member_ids:
            if eid in old_entity_community:
                old_cids.add(old_entity_community[eid])

        cid = graph._next_community_id
        graph._next_community_id += 1

        key_concepts = []
        for eid in sorted(
            member_ids,
            key=lambda x: graph.entities[x].mentions,
            reverse=True,
        )[:5]:
            key_concepts.append(graph.entities[eid].name)

        if len(old_cids) == 1:
            old_cid = next(iter(old_cids))
            old_comm = old_communities.get(old_cid)
            if old_comm and set(old_comm.entity_ids) == set(member_ids):
                community = Community(
                    id=cid,
                    entity_ids=member_ids,
                    summary=old_comm.summary,
                    key_concepts=key_concepts,
                    dirty=old_comm.dirty,
                    entity_count_at_last_summary=old_comm.entity_count_at_last_summary,
                    last_updated=old_comm.last_updated,
                )
            else:
                community = Community(
                    id=cid,
                    entity_ids=member_ids,
                    summary=old_comm.summary if old_comm else "",
                    key_concepts=key_concepts,
                    dirty=True,
                    entity_count_at_last_summary=old_comm.entity_count_at_last_summary if old_comm else 0,
                )
        else:
            community = Community(
                id=cid,
                entity_ids=member_ids,
                key_concepts=key_concepts,
                dirty=True,
            )

        graph.communities[cid] = community
        for eid in member_ids:
            graph.entities[eid].community_id = cid


def _resummarize_dirty(llm: LLMProvider, graph: GraphStore) -> int:
    """Re-summarize only dirty communities using Pro."""
    count = 0
    for community in graph.communities.values():
        if not community.needs_resummarization:
            continue

        if len(community.entity_ids) < 2:
            if community.entity_ids:
                e = graph.entities[community.entity_ids[0]]
                community.summary = f"{e.name}: {e.description}"
            community.dirty = False
            community.entity_count_at_last_summary = len(community.entity_ids)
            community.last_updated = datetime.datetime.now().isoformat()
            continue

        entities_text: list[str] = []
        for eid in community.entity_ids[:20]:
            e = graph.entities.get(eid)
            if e:
                entities_text.append(f"  - {e.name} ({e.entity_type}): {e.description}")

        rels_text: list[str] = []
        member_set = set(community.entity_ids)
        for rel in graph.relationships:
            if rel.source_id in member_set and rel.target_id in member_set:
                src = graph.entities.get(rel.source_id)
                tgt = graph.entities.get(rel.target_id)
                if src and tgt:
                    rels_text.append(
                        f"  - {src.name} --[{rel.relation_type}]--> {tgt.name}"
                        f"{': ' + rel.description if rel.description else ''}"
                    )

        from loom.prompts import COMMUNITY_SUMMARY
        prompt = COMMUNITY_SUMMARY.format(
            entities="\n".join(entities_text[:20]),
            relationships="\n".join(rels_text[:20]),
        )

        try:
            resp = llm.generate(prompt, model="flash", temperature=0.3, max_output_tokens=512)
            community.summary = resp.text.strip()
        except Exception as e:
            community.summary = f"Cluster: {', '.join(community.key_concepts)}"
            print(f"  [Hierarchy] Community summary failed: {e}")

        community.dirty = False
        community.entity_count_at_last_summary = len(community.entity_ids)
        community.last_updated = datetime.datetime.now().isoformat()
        count += 1

    return count
