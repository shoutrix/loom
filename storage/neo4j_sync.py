"""
Neo4j write-through persistence with WAL replay.

The in-memory graph is primary. Neo4j is the durable backup.
Sync worker reads unsynced WAL entries and applies them to Neo4j.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from loom.config import Neo4jSettings
    from loom.graph.store import GraphStore


class Neo4jSync:
    """Manages bidirectional sync between in-memory graph and Neo4j."""

    def __init__(self, settings: Neo4jSettings) -> None:
        self.settings = settings
        self._driver = None
        self._connected = False

    def connect(self) -> bool:
        if not self.settings.enabled or not self.settings.uri:
            return False
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.settings.uri,
                auth=(self.settings.username, self.settings.password),
            )
            self._driver.verify_connectivity()
            self._connected = True
            self._ensure_indexes()
            return True
        except Exception as e:
            print(f"  [Neo4j] Connection failed: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        if self._driver:
            self._driver.close()
            self._connected = False

    def _ensure_indexes(self) -> None:
        if not self._connected or not self._driver:
            return
        with self._driver.session(database=self.settings.database) as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Community) ON (c.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (b:Bridge) ON (b.entity_id)")

    def sync_wal(self, graph: GraphStore) -> int:
        """Replay unsynced WAL entries to Neo4j. Returns count of synced entries."""
        if not self._connected or not self._driver or not graph.wal:
            return 0

        entries = graph.wal.unsynced_entries()
        if not entries:
            return 0

        synced = 0
        batch: list[dict[str, Any]] = []

        for entry in entries:
            batch.append(entry)
            if len(batch) >= 100:
                self._apply_batch(batch)
                synced += len(batch)
                batch = []

        if batch:
            self._apply_batch(batch)
            synced += len(batch)

        return synced

    def _apply_batch(self, entries: list[dict[str, Any]]) -> None:
        if not self._driver:
            return
        with self._driver.session(database=self.settings.database) as session:
            for entry in entries:
                op = entry.get("op", "")
                payload = entry.get("payload", {})
                try:
                    if op == "add_entity":
                        session.run(
                            "MERGE (e:Entity {id: $id}) "
                            "SET e.name = $name, e.type = $type, "
                            "e.description = $description, e.mentions = $mentions",
                            id=payload.get("id"),
                            name=payload.get("name"),
                            type=payload.get("type"),
                            description=payload.get("description"),
                            mentions=payload.get("mentions", 1),
                        )
                    elif op == "add_relationship":
                        session.run(
                            "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                            "MERGE (a)-[r:RELATES {type: $type}]->(b) "
                            "SET r.description = $desc, r.weight = $weight",
                            src=payload.get("source_id"),
                            tgt=payload.get("target_id"),
                            type=payload.get("type"),
                            desc=payload.get("description"),
                            weight=payload.get("weight", 1.0),
                        )
                    elif op == "merge_entity":
                        session.run(
                            "MATCH (e:Entity {id: $id}) "
                            "SET e.mentions = e.mentions + 1",
                            id=payload.get("existing_id"),
                        )
                except Exception as e:
                    print(f"  [Neo4j] Failed to apply {op}: {e}")

    def sync_full_graph(self, graph: GraphStore) -> dict[str, int]:
        """Push the complete in-memory graph state to Neo4j (communities, bridges, latent connections)."""
        if not self._connected or not self._driver:
            return {}
        counts: dict[str, int] = {"communities": 0, "bridges": 0, "latent_connections": 0}

        with self._driver.session(database=self.settings.database) as session:
            for c in graph.communities.values():
                session.run(
                    "MERGE (c:Community {id: $id}) "
                    "SET c.summary = $summary, c.key_concepts = $key_concepts, "
                    "c.entity_count = $entity_count, c.dirty = $dirty",
                    id=c.id,
                    summary=c.summary,
                    key_concepts=c.key_concepts,
                    entity_count=len(c.entity_ids),
                    dirty=c.dirty,
                )
                for eid in c.entity_ids:
                    session.run(
                        "MATCH (e:Entity {id: $eid}), (c:Community {id: $cid}) "
                        "MERGE (e)-[:MEMBER_OF]->(c)",
                        eid=eid, cid=c.id,
                    )
                counts["communities"] += 1

            for b in graph.bridges.values():
                session.run(
                    "MERGE (b:Bridge {entity_id: $eid}) "
                    "SET b.novelty_score = $score, b.source_doc_ids = $docs",
                    eid=b.entity_id,
                    score=b.novelty_score,
                    docs=b.source_doc_ids,
                )
                session.run(
                    "MATCH (e:Entity {id: $eid}), (b:Bridge {entity_id: $eid}) "
                    "MERGE (e)-[:IS_BRIDGE]->(b)",
                    eid=b.entity_id,
                )
                counts["bridges"] += 1

            for lc in graph.latent_connections:
                session.run(
                    "MATCH (a:Entity {id: $aid}), (b:Entity {id: $bid}) "
                    "MERGE (a)-[r:LATENT_CONNECTION]->(b) "
                    "SET r.similarity = $sim, r.confirmed = $conf, "
                    "r.description = $desc, r.novelty_score = $novelty",
                    aid=lc.entity_a_id, bid=lc.entity_b_id,
                    sim=lc.similarity, conf=lc.confirmed,
                    desc=lc.description, novelty=lc.novelty_score,
                )
                counts["latent_connections"] += 1

        return counts

    def load_full_graph(self, graph: GraphStore) -> int:
        """Load the full graph from Neo4j into memory. Returns entity count."""
        if not self._connected or not self._driver:
            return 0

        from loom.graph.models import Entity, Relationship, Community, BridgeEdge, LatentConnection

        count = 0
        with self._driver.session(database=self.settings.database) as session:
            result = session.run("MATCH (e:Entity) RETURN e")
            for record in result:
                node = record["e"]
                entity = Entity(
                    id=node["id"],
                    name=node.get("name", ""),
                    entity_type=node.get("type", "concept"),
                    description=node.get("description", ""),
                    mentions=node.get("mentions", 1),
                )
                graph.entities[entity.id] = entity
                graph._name_index[entity.name.strip().lower()] = entity.id
                count += 1

            result = session.run(
                "MATCH (a:Entity)-[r:RELATES]->(b:Entity) "
                "RETURN a.id AS src, b.id AS tgt, r.type AS type, "
                "r.description AS desc, r.weight AS weight"
            )
            for record in result:
                rel = Relationship(
                    source_id=record["src"],
                    target_id=record["tgt"],
                    relation_type=record.get("type", "related_to"),
                    description=record.get("desc", ""),
                    weight=record.get("weight", 1.0),
                )
                graph.relationships.append(rel)
                graph._adjacency[rel.source_id].add(rel.target_id)
                graph._adjacency[rel.target_id].add(rel.source_id)

            result = session.run("MATCH (c:Community) RETURN c")
            for record in result:
                node = record["c"]
                cid = node["id"]
                members_result = session.run(
                    "MATCH (e:Entity)-[:MEMBER_OF]->(c:Community {id: $cid}) RETURN e.id AS eid",
                    cid=cid,
                )
                entity_ids = [r["eid"] for r in members_result]
                community = Community(
                    id=cid,
                    entity_ids=entity_ids,
                    summary=node.get("summary", ""),
                    key_concepts=list(node.get("key_concepts", [])),
                    dirty=node.get("dirty", False),
                )
                graph.communities[community.id] = community

            result = session.run("MATCH (b:Bridge) RETURN b")
            for record in result:
                node = record["b"]
                bridge = BridgeEdge(
                    entity_id=node["entity_id"],
                    source_doc_ids=list(node.get("source_doc_ids", [])),
                    novelty_score=node.get("novelty_score", 0.0),
                )
                graph.bridges[bridge.entity_id] = bridge

            result = session.run(
                "MATCH (a:Entity)-[r:LATENT_CONNECTION]->(b:Entity) "
                "RETURN a.id AS aid, b.id AS bid, r.similarity AS sim, "
                "r.confirmed AS conf, r.description AS desc, r.novelty_score AS novelty"
            )
            for record in result:
                lc = LatentConnection(
                    entity_a_id=record["aid"],
                    entity_b_id=record["bid"],
                    similarity=record.get("sim", 0.0),
                    confirmed=record.get("conf", False),
                    description=record.get("desc", ""),
                    novelty_score=record.get("novelty", 0.0),
                )
                graph.latent_connections.append(lc)

        return count
