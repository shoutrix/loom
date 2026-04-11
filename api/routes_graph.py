"""GET /graph -- explore the knowledge graph."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/graph", tags=["graph"])


class GraphStatsResponse(BaseModel):
    entities: int
    relationships: int
    communities: int
    bridges: int
    latent_connections: int


class EntityResponse(BaseModel):
    id: str
    name: str
    type: str
    description: str
    mentions: int
    confidence: float
    community_id: int
    source_doc_ids: list[str]


class CommunityResponse(BaseModel):
    id: int
    size: int
    key_concepts: list[str]
    summary: str
    dirty: bool


@router.get("/stats", response_model=GraphStatsResponse)
async def graph_stats() -> GraphStatsResponse:
    from loom.main import get_app_state
    state = get_app_state()
    s = state.graph.stats()
    return GraphStatsResponse(**s)


@router.get("/entities", response_model=list[EntityResponse])
async def list_entities(
    min_confidence: float = Query(0.0),
    limit: int = Query(100),
) -> list[EntityResponse]:
    from loom.main import get_app_state
    state = get_app_state()

    entities = sorted(
        state.graph.entities.values(),
        key=lambda e: e.confidence,
        reverse=True,
    )

    return [
        EntityResponse(
            id=e.id,
            name=e.name,
            type=e.entity_type,
            description=e.description,
            mentions=e.mentions,
            confidence=round(e.confidence, 3),
            community_id=e.community_id,
            source_doc_ids=e.source_doc_ids,
        )
        for e in entities
        if e.confidence >= min_confidence
    ][:limit]


@router.get("/entity/{entity_id}")
async def get_entity(entity_id: str) -> dict:
    from loom.main import get_app_state
    state = get_app_state()

    entity = state.graph.entities.get(entity_id)
    if not entity:
        return {"error": "Entity not found"}

    relationships = state.graph.get_entity_relationships(entity_id)

    return {
        "entity": entity.to_dict(),
        "relationships": [r.to_dict() for r in relationships],
    }


@router.get("/communities", response_model=list[CommunityResponse])
async def list_communities() -> list[CommunityResponse]:
    from loom.main import get_app_state
    state = get_app_state()

    return [
        CommunityResponse(
            id=c.id,
            size=len(c.entity_ids),
            key_concepts=c.key_concepts,
            summary=c.summary,
            dirty=c.dirty,
        )
        for c in sorted(
            state.graph.communities.values(),
            key=lambda c: len(c.entity_ids),
            reverse=True,
        )
    ]


@router.get("/bridges")
async def list_bridges() -> list[dict]:
    from loom.main import get_app_state
    state = get_app_state()

    bridges = []
    for bridge in state.graph.bridges.values():
        entity = state.graph.entities.get(bridge.entity_id)
        if entity:
            bridges.append({
                "entity": entity.to_dict(),
                "source_doc_ids": bridge.source_doc_ids,
                "novelty_score": bridge.novelty_score,
            })
    return bridges


@router.get("/connections")
async def list_connections() -> list[dict]:
    from loom.main import get_app_state
    state = get_app_state()

    return [lc.to_dict() for lc in state.graph.latent_connections if lc.confirmed]


@router.get("/export")
async def export_graph() -> dict:
    from loom.main import get_app_state
    state = get_app_state()
    return state.graph.to_dict()


@router.post("/sync-neo4j")
async def sync_neo4j() -> dict:
    """Manually trigger WAL replay to Neo4j."""
    from loom.main import get_app_state
    state = get_app_state()

    if not state.neo4j_sync:
        return {"status": "disabled", "message": "Neo4j sync is not enabled"}

    synced = state.neo4j_sync.sync_wal(state.graph)
    return {"status": "ok", "entries_synced": synced}
