"""
Chat engine: hybrid retrieval + context assembly + Pro answer.

Uses Pro for multi-hop reasoning over retrieved context.
Maintains conversation memory for follow-up questions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from loom.config import Settings
    from loom.llm.provider import LLMProvider
    from loom.llm.embeddings import EmbeddingProvider
    from loom.search.semantic import DualSemanticIndex
    from loom.search.keyword import KeywordIndex
    from loom.graph.store import GraphStore

from loom.prompts import CHAT_SYSTEM, CHAT_ANSWER

SYSTEM_PROMPT = CHAT_SYSTEM


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    sources: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ChatResponse:
    answer: str
    sources: list[dict[str, Any]]
    graph_context: str
    num_chunks_retrieved: int
    num_propositions_retrieved: int


class ChatEngine:
    """Conversational Q&A over the knowledge base."""

    def __init__(
        self,
        settings: Settings,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        semantic_index: DualSemanticIndex,
        keyword_index: KeywordIndex,
        graph: GraphStore,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.embedder = embedder
        self.semantic_index = semantic_index
        self.keyword_index = keyword_index
        self.graph = graph
        self.history: list[ChatMessage] = []
        self.max_history = 10

    def chat(self, user_message: str) -> ChatResponse:
        """Process a user message and return an answer with sources."""
        from loom.search.hybrid import hybrid_search

        self.history.append(ChatMessage(role="user", content=user_message))

        context_query = user_message
        if len(self.history) >= 3:
            recent = self.history[-3:]
            context_query = " ".join(m.content for m in recent if m.role == "user")

        results, graph_context = hybrid_search(
            context_query,
            self.embedder,
            self.semantic_index,
            self.keyword_index,
            self.graph,
            self.settings.search,
        )

        chunks = [r for r in results if not r.is_proposition]
        props = [r for r in results if r.is_proposition]

        context_parts: list[str] = []

        if graph_context:
            context_parts.append(graph_context)

        if props:
            context_parts.append("\n=== RELEVANT FACTS (precise matches) ===")
            for r in props[:15]:
                context_parts.append(f"[{r.doc_id}] {r.text}")

        if chunks:
            context_parts.append("\n=== RELEVANT PASSAGES (broader context) ===")
            for r in chunks[:10]:
                preview = r.text[:500]
                context_parts.append(f"[{r.doc_id}]\n{preview}")

        full_context = "\n\n".join(context_parts)

        conversation_text = ""
        if len(self.history) > 1:
            recent = self.history[-(self.max_history):]
            conv_lines = []
            for msg in recent[:-1]:
                role = "User" if msg.role == "user" else "Assistant"
                conv_lines.append(f"{role}: {msg.content[:200]}")
            conversation_text = "\n".join(conv_lines)

        prompt = self._build_prompt(user_message, full_context, conversation_text)

        resp = self.llm.generate(
            prompt,
            model="pro",
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=4096,
        )

        sources = [
            {"id": r.id, "doc_id": r.doc_id, "text": r.text[:200], "score": round(r.score, 3)}
            for r in results[:10]
        ]

        answer_msg = ChatMessage(role="assistant", content=resp.text, sources=sources)
        self.history.append(answer_msg)

        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history:]

        return ChatResponse(
            answer=resp.text,
            sources=sources,
            graph_context=graph_context,
            num_chunks_retrieved=len(chunks),
            num_propositions_retrieved=len(props),
        )

    def clear_history(self) -> None:
        self.history.clear()

    def save_history(self, path: Path) -> None:
        """Persist chat history to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"role": m.role, "content": m.content, "sources": m.sources}
            for m in self.history
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_history(self, path: Path) -> bool:
        """Reload chat history from disk. Returns True on success."""
        if not path.exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self.history = [
                ChatMessage(role=m["role"], content=m["content"], sources=m.get("sources", []))
                for m in data
            ]
            return True
        except Exception:
            return False

    def _build_prompt(
        self,
        question: str,
        context: str,
        conversation: str,
    ) -> str:
        parts = []

        if conversation:
            parts.append(f"Previous conversation:\n{conversation}\n")

        parts.append(f"Retrieved context:\n{context}\n")
        parts.append(f"User question: {question}")
        parts.append(f"\n{CHAT_ANSWER}")

        return "\n\n".join(parts)
