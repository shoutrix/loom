"""
In-memory graph store with WAL, name_index, and entity embedding index.

The in-memory graph is the primary data store. Neo4j is the durable
backup (optional). All reads hit local memory.
"""

from __future__ import annotations

import json
import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from loom.graph.models import (
    Entity,
    Relationship,
    Community,
    BridgeEdge,
    LatentConnection,
    MergeLogEntry,
)


class WAL:
    """Write-Ahead Log for graph mutations."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._seq = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with open(self.path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        self._seq = max(self._seq, entry.get("seq", 0))
                    except json.JSONDecodeError:
                        continue

    def append(self, op: str, payload: dict[str, Any]) -> int:
        self._seq += 1
        entry = {
            "seq": self._seq,
            "op": op,
            "payload": payload,
            "synced": False,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return self._seq

    def unsynced_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if not self.path.exists():
            return entries
        with open(self.path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if not entry.get("synced", False):
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
        return entries


class GraphStore:
    """In-memory knowledge graph with WAL persistence."""

    def __init__(self, wal_path: Path | None = None) -> None:
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        self.communities: dict[int, Community] = {}
        self.bridges: dict[str, BridgeEdge] = {}
        self.latent_connections: list[LatentConnection] = []
        self.merge_log: list[MergeLogEntry] = []

        self._name_index: dict[str, str] = {}  # normalized_name -> entity_id
        self._next_community_id: int = 0
        self._adjacency: dict[str, set[str]] = defaultdict(set)

        self.wal = WAL(wal_path) if wal_path else None

    def add_entity(self, entity: Entity) -> Entity:
        """Add an entity to the graph."""
        self.entities[entity.id] = entity
        normalized = entity.name.strip().lower()
        self._name_index[normalized] = entity.id

        if self.wal:
            self.wal.append("add_entity", entity.to_dict())
        return entity

    def get_entity_by_name(self, name: str) -> Entity | None:
        normalized = name.strip().lower()
        eid = self._name_index.get(normalized)
        if eid:
            return self.entities.get(eid)
        return None

    def merge_entity(
        self,
        existing_id: str,
        new_entity: Entity,
        reason: str = "",
    ) -> Entity:
        """Merge a new entity into an existing one."""
        existing = self.entities[existing_id]
        existing.mentions += new_entity.mentions

        for doc_id in new_entity.source_doc_ids:
            if doc_id not in existing.source_doc_ids:
                existing.source_doc_ids.append(doc_id)

        if new_entity.description and new_entity.description != existing.description:
            if len(new_entity.description) > len(existing.description):
                existing.description = new_entity.description

        existing.updated_at = datetime.datetime.now().isoformat()

        new_normalized = new_entity.name.strip().lower()
        self._name_index[new_normalized] = existing_id

        self.merge_log.append(MergeLogEntry(
            merged_from_id=new_entity.id,
            merged_into_id=existing_id,
            merged_from_name=new_entity.name,
            merged_into_name=existing.name,
            reason=reason,
            timestamp=datetime.datetime.now().isoformat(),
        ))

        if existing.community_id >= 0 and existing.community_id in self.communities:
            self.communities[existing.community_id].dirty = True

        if self.wal:
            self.wal.append("merge_entity", {
                "existing_id": existing_id,
                "merged_name": new_entity.name,
                "reason": reason,
            })

        return existing

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        description: str = "",
        source_doc_id: str = "",
        weight: float = 1.0,
    ) -> Relationship | None:
        if source_id not in self.entities or target_id not in self.entities:
            return None
        if source_id == target_id:
            return None

        for existing in self.relationships:
            if (
                existing.source_id == source_id
                and existing.target_id == target_id
                and existing.relation_type == relation_type
            ):
                existing.weight += 0.5
                return existing

        rel = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            description=description,
            weight=weight,
            source_doc_id=source_doc_id,
            created_at=datetime.datetime.now().isoformat(),
        )
        self.relationships.append(rel)
        self._adjacency[source_id].add(target_id)
        self._adjacency[target_id].add(source_id)

        if self.wal:
            self.wal.append("add_relationship", rel.to_dict())

        return rel

    def get_entity_relationships(self, entity_id: str) -> list[Relationship]:
        return [
            r for r in self.relationships
            if r.source_id == entity_id or r.target_id == entity_id
        ]

    def get_entity_neighbors(self, entity_id: str) -> list[str]:
        return list(self._adjacency.get(entity_id, set()))

    def shortest_path_length(self, start_id: str, end_id: str, max_depth: int = 10) -> int:
        """BFS shortest path. Returns -1 if no path found."""
        if start_id == end_id:
            return 0
        if start_id not in self._adjacency or end_id not in self._adjacency:
            return -1

        visited: set[str] = {start_id}
        queue: list[tuple[str, int]] = [(start_id, 0)]
        head = 0

        while head < len(queue):
            current, depth = queue[head]
            head += 1
            if depth >= max_depth:
                continue
            for neighbor in self._adjacency.get(current, set()):
                if neighbor == end_id:
                    return depth + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return -1

    def get_all_entity_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Return all entity IDs and their embeddings as a matrix."""
        ids: list[str] = []
        vecs: list[np.ndarray] = []
        for eid, entity in self.entities.items():
            if entity.embedding is not None:
                ids.append(eid)
                vecs.append(entity.embedding)
        if not vecs:
            return [], np.empty((0, 768), dtype=np.float32)
        return ids, np.stack(vecs)

    def stats(self) -> dict[str, Any]:
        return {
            "entities": len(self.entities),
            "relationships": len(self.relationships),
            "communities": len(self.communities),
            "bridges": len(self.bridges),
            "latent_connections": len(self.latent_connections),
            "merge_log_entries": len(self.merge_log),
        }

    def to_dict(self, include_embeddings: bool = True) -> dict[str, Any]:
        return {
            "entities": [e.to_dict(include_embedding=include_embeddings) for e in self.entities.values()],
            "relationships": [r.to_dict() for r in self.relationships],
            "communities": [c.to_dict() for c in self.communities.values()],
            "bridges": [b.to_dict() for b in self.bridges.values()],
            "latent_connections": [lc.to_dict() for lc in self.latent_connections],
            "merge_log": [
                {"from": m.merged_from_name, "into": m.merged_into_name, "reason": m.reason, "ts": m.timestamp}
                for m in self.merge_log
            ],
            "stats": self.stats(),
        }

    def save_snapshot(self, path: Path) -> None:
        """Save full graph state to a JSON snapshot (including embeddings)."""
        data = self.to_dict(include_embeddings=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_snapshot(self, path: Path) -> None:
        """Load graph state from a JSON snapshot (including embeddings)."""
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)

        for e_dict in data.get("entities", []):
            embedding = None
            if "embedding_b64" in e_dict and "embedding_dim" in e_dict:
                try:
                    embedding = Entity.embedding_from_b64(e_dict["embedding_b64"], e_dict["embedding_dim"])
                except Exception:
                    pass

            entity = Entity(
                id=e_dict["id"],
                name=e_dict["name"],
                entity_type=e_dict.get("type", "concept"),
                description=e_dict.get("description", ""),
                mentions=e_dict.get("mentions", 1),
                source_doc_ids=e_dict.get("source_doc_ids", []),
                community_id=e_dict.get("community_id", -1),
                embedding_model=e_dict.get("embedding_model", ""),
                embedding=embedding,
            )
            self.entities[entity.id] = entity
            self._name_index[entity.name.strip().lower()] = entity.id

        for r_dict in data.get("relationships", []):
            rel = Relationship(
                id=r_dict.get("id", ""),
                source_id=r_dict["source_id"],
                target_id=r_dict["target_id"],
                relation_type=r_dict.get("type", "related_to"),
                description=r_dict.get("description", ""),
                weight=r_dict.get("weight", 1.0),
                source_doc_id=r_dict.get("source_doc_id", ""),
            )
            self.relationships.append(rel)
            self._adjacency[rel.source_id].add(rel.target_id)
            self._adjacency[rel.target_id].add(rel.source_id)

        for c_dict in data.get("communities", []):
            community = Community(
                id=c_dict["id"],
                entity_ids=c_dict.get("entity_ids", []),
                summary=c_dict.get("summary", ""),
                key_concepts=c_dict.get("key_concepts", []),
                dirty=c_dict.get("dirty", False),
                entity_count_at_last_summary=c_dict.get("entity_count_at_last_summary", 0),
            )
            self.communities[community.id] = community
            self._next_community_id = max(self._next_community_id, community.id + 1)

        for b_dict in data.get("bridges", []):
            bridge = BridgeEdge(
                entity_id=b_dict["entity_id"],
                source_doc_ids=b_dict.get("source_doc_ids", []),
                novelty_score=b_dict.get("novelty_score", 0.0),
            )
            self.bridges[bridge.entity_id] = bridge

        for lc_dict in data.get("latent_connections", []):
            lc = LatentConnection(
                entity_a_id=lc_dict["entity_a_id"],
                entity_b_id=lc_dict["entity_b_id"],
                similarity=lc_dict.get("similarity", 0.0),
                confirmed=lc_dict.get("confirmed", False),
                description=lc_dict.get("description", ""),
                novelty_score=lc_dict.get("novelty_score", 0.0),
            )
            self.latent_connections.append(lc)
