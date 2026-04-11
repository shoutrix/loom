"""POST /search -- hybrid search over the knowledge base."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20


class SearchResultItem(BaseModel):
    id: str
    text: str
    score: float
    doc_id: str
    is_proposition: bool
    source: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    graph_context: str
    total_results: int


@router.post("", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    from loom.main import get_app_state
    from loom.search.hybrid import hybrid_search
    state = get_app_state()

    results, graph_context = hybrid_search(
        req.query,
        state.embedder,
        state.semantic_index,
        state.keyword_index,
        state.graph,
        state.settings.search,
    )

    items = [
        SearchResultItem(
            id=r.id,
            text=r.text[:500],
            score=round(r.score, 4),
            doc_id=r.doc_id,
            is_proposition=r.is_proposition,
            source=r.source,
        )
        for r in results[:req.top_k]
    ]

    return SearchResponse(
        results=items,
        graph_context=graph_context[:2000],
        total_results=len(results),
    )
