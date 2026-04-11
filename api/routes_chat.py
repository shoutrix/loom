"""POST /chat -- Q&A with sources."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str


class SourceItem(BaseModel):
    id: str
    doc_id: str
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    num_chunks_retrieved: int
    num_propositions_retrieved: int


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    from loom.main import get_app_state
    state = get_app_state()

    result = state.chat_engine.chat(req.message)

    sources = [
        SourceItem(
            id=s.get("id", ""),
            doc_id=s.get("doc_id", ""),
            text=s.get("text", "")[:300],
            score=s.get("score", 0.0),
        )
        for s in result.sources
    ]

    return ChatResponse(
        answer=result.answer,
        sources=sources,
        num_chunks_retrieved=result.num_chunks_retrieved,
        num_propositions_retrieved=result.num_propositions_retrieved,
    )


@router.post("/clear")
async def clear_chat() -> dict:
    from loom.main import get_app_state
    state = get_app_state()
    state.chat_engine.clear_history()
    return {"status": "cleared"}
