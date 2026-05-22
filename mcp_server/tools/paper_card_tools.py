"""
Paper-card MCP tools.

`submit_paper_card` lets the calling agent (Claude / Cursor) hand a
pre-built structured review directly to loom. The card is persisted
as-is, bypassing the Gemini-based extractor the FastAPI server would
otherwise run on first UI view.

Workflow this enables:

    1. Agent reads a paper end-to-end.
    2. Agent calls ingest_paper(ws, identifier)
       → loom runs the full ingestion pipeline (chunking, embeddings,
         graph extraction) via MCP sampling.
    3. Agent calls submit_paper_card(ws, paper_id, card)
       → loom persists the structured review without calling its own
         LLM. The UI renders it instantly when the user opens the paper.

The schema in the docstring is intentionally verbose — it's what the
MCP host's LLM (Claude) reads to know what to produce. The handler
itself is tolerant: missing fields default to empty, extra fields are
preserved, type-mismatched values are coerced via the same helpers
the Gemini extractor uses.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from loom.mcp_server.state import MCPState
from loom.mcp_server.workspace import MCPWorkspaceLoader
from loom.paper_card import PaperCard, Dataset, RelatedWork, save_card, load_card
from loom.paper_card.extractor import (
    _coerce_datasets,
    _coerce_related_work,
    _coerce_str_list,
)
from loom.permissions import enforce


def _header_from_registry(registry, paper_id: str) -> dict[str, Any]:
    """Build the header dict (title/authors/venue/year/source_url/...) from
    the workspace's paper registry. Anything missing returns "" / None /
    empty list — the agent's card can override these.
    """
    rec = registry.get(paper_id) if registry else None
    if rec is None:
        return {}

    arxiv_id = getattr(rec, "arxiv_id", "") or ""
    doi = getattr(rec, "doi", "") or ""
    source_url = ""
    if arxiv_id:
        source_url = f"https://arxiv.org/abs/{arxiv_id}"
    elif doi:
        source_url = f"https://doi.org/{doi}"

    return {
        "title": getattr(rec, "title", "") or "",
        "authors": list(getattr(rec, "authors", []) or []),
        "venue": getattr(rec, "venue", "") or "",
        "year": getattr(rec, "year", None) if isinstance(getattr(rec, "year", None), int) else None,
        "source_url": source_url,
        "arxiv_id": arxiv_id,
        "doi": doi,
    }


def _coerce_card_input(
    paper_id: str,
    card_in: dict[str, Any],
    *,
    header_defaults: dict[str, Any],
    model_tag: str = "agent-submitted",
) -> PaperCard:
    """Build a PaperCard from the agent's JSON input.

    Header fields fall back to the registry-derived defaults if the
    agent didn't supply them. List fields go through the same coercion
    used by the Gemini extractor so we tolerate strings-where-objects
    were expected and similar minor shape variations.
    """
    import datetime

    def _str(value, default: str = "") -> str:
        return str(value).strip() if isinstance(value, (str, int, float)) else default

    def _maybe_int(value):
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    def _str_list(value) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for v in value:
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
            elif isinstance(v, (int, float)):
                out.append(str(v))
        return out

    header_in = card_in.get("header") if isinstance(card_in.get("header"), dict) else {}
    title = _str(card_in.get("title")) or _str(header_in.get("title")) or _str(header_defaults.get("title"))
    authors = _str_list(card_in.get("authors")) or _str_list(header_in.get("authors")) or _str_list(header_defaults.get("authors"))
    venue = _str(card_in.get("venue")) or _str(header_in.get("venue")) or _str(header_defaults.get("venue"))
    year = _maybe_int(card_in.get("year")) or _maybe_int(header_in.get("year"))
    if year is None:
        year = header_defaults.get("year") if isinstance(header_defaults.get("year"), int) else None
    source_url = _str(card_in.get("source_url")) or _str(header_in.get("source_url")) or _str(header_defaults.get("source_url"))
    arxiv_id = _str(card_in.get("arxiv_id")) or _str(header_in.get("arxiv_id")) or _str(header_defaults.get("arxiv_id"))
    doi = _str(card_in.get("doi")) or _str(header_in.get("doi")) or _str(header_defaults.get("doi"))

    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    return PaperCard(
        version=1,
        generated_at=now,
        model=model_tag,
        paper_id=paper_id,
        title=title,
        authors=authors,
        venue=venue,
        year=year,
        source_url=source_url,
        arxiv_id=arxiv_id,
        doi=doi,
        tldr=_str(card_in.get("tldr")),
        problem=_str(card_in.get("problem")),
        approach=_str(card_in.get("approach")),
        contributions=_coerce_str_list(card_in.get("contributions")),
        datasets=_coerce_datasets(card_in.get("datasets")),
        setup=_str(card_in.get("setup")),
        results=_coerce_str_list(card_in.get("results")),
        conclusion=_str(card_in.get("conclusion")),
        strengths=_coerce_str_list(card_in.get("strengths")),
        limitations=_coerce_str_list(card_in.get("limitations")),
        related_work=_coerce_related_work(card_in.get("related_work")),
        workspace_relevance=_str(card_in.get("workspace_relevance")),
        open_questions=_coerce_str_list(card_in.get("open_questions")),
    )


def register(mcp: FastMCP, state: MCPState, loader: MCPWorkspaceLoader) -> None:

    @mcp.tool()
    async def submit_paper_card(
        workspace_id: str,
        paper_id: str,
        card: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Persist a pre-built structured review for a paper, bypassing loom's
        Gemini-based extractor.

        WHEN TO CALL THIS
        =================
        Immediately after ingest_paper returns successfully. You already
        read the paper to ingest it, so producing the card is cheap. If
        you skip this, loom will run its own extractor (one Gemini call)
        the first time the user opens the paper in the UI — duplicating
        work you already did.

        You MAY also call this for an existing paper at any later time —
        the card will be overwritten.

        ARGUMENTS
        =========
        - workspace_id: target workspace (must be writable for this
          subscriber).
        - paper_id: id from the workspace's paper_registry (returned by
          ingest_paper). The paper does not have to be `status='ingested'`
          yet — you can attach a card to a paper that's still ingesting.
        - card: a JSON object matching the schema below.

        REQUIRED CARD SCHEMA (13 review fields)
        ========================================
        All fields except `tldr` may be empty if the paper genuinely
        doesn't support them — but produce non-empty content wherever
        you can. The UI renders empty fields by hiding them, so leaving
        a field blank silently degrades the user's view.

        {
          "tldr": "<1-2 sentences. What this paper is and why it matters.>",
          "problem": "<2-4 sentences. The specific problem being solved.>",
          "approach": "<3-6 sentences. The method, architecture, or
              algorithm — concrete enough that another researcher could
              sketch it.>",
          "contributions": [
              "<bullet — distinct claim or artifact this paper produces>",
              ...
          ],
          "datasets": [
              {"name": "<dataset name>",
               "size": "<e.g. '278 tasks', '50k examples', '1M tokens'>",
               "type": "<e.g. 'speech', 'TOD', 'image', 'code'>"}
          ],
          "setup": "<2-4 sentences. Baselines, metrics, evaluation setup,
              hardware if mentioned.>",
          "results": [
              "<headline result with concrete numbers from the paper —
                accuracy %, BLEU, MOS, citation count, etc.>",
              ...
          ],
          "conclusion": "<2-3 sentences. What did the paper show.>",
          "strengths": ["<bullet>", ...],
          "limitations": [
              "<bullet — include both author-acknowledged and reviewer-
                identified limitations>",
              ...
          ],
          "related_work": [
              {"title": "<canonical paper title>",
               "why": "<1 line: why this paper matters to the target>"}
          ],
          "workspace_relevance": "<2-3 sentences. Why this paper matters
              for THIS workspace specifically. If you don't know the
              workspace's focus, frame it as why the paper is generally
              interesting.>",
          "open_questions": ["<bullet>", ...]
        }

        OPTIONAL HEADER OVERRIDES
        =========================
        These fall back to the registry record if you omit them. Pass
        them explicitly when the registry is incomplete (e.g. the
        identifier was a URL and we don't know the venue):

          title, authors (list[str]), venue, year (int),
          source_url, arxiv_id, doi

        EXTRA FIELDS
        ============
        Anything beyond the schema is silently ignored. Don't rely on
        custom fields surviving — the UI only renders the 13 review
        fields above.

        RETURNS
        =======
        {ok: bool, paper_id: str, path: str, model: str, fields_filled: int}

        On failure (paper_id not in registry, permission denied):
        {ok: false, error: "..."}
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err

        ws_settings = state.settings.for_workspace(workspace_id)

        # Try to read the registry header so we can fill in missing
        # bibliographic fields. If the paper isn't in the registry at all,
        # we still accept the card — the agent might be submitting one
        # for a paper they're about to ingest.
        try:
            ws = loader.load(workspace_id)
            registry = ws.registry
        except Exception:
            registry = None

        header_defaults = _header_from_registry(registry, paper_id)

        if not isinstance(card, dict):
            return {
                "ok": False,
                "error": (
                    f"`card` must be a JSON object, got {type(card).__name__}"
                ),
            }

        built = _coerce_card_input(paper_id, card, header_defaults=header_defaults)
        path = save_card(ws_settings.data_dir, built)

        fields_filled = sum([
            bool(built.tldr),
            bool(built.problem),
            bool(built.approach),
            len(built.contributions) > 0,
            len(built.datasets) > 0,
            bool(built.setup),
            len(built.results) > 0,
            bool(built.conclusion),
            len(built.strengths) > 0,
            len(built.limitations) > 0,
            len(built.related_work) > 0,
            bool(built.workspace_relevance),
            len(built.open_questions) > 0,
        ])

        return {
            "ok": True,
            "paper_id": paper_id,
            "path": str(path),
            "model": built.model,
            "fields_filled": fields_filled,
            "total_fields": 13,
        }

    @mcp.tool()
    async def get_paper_card(
        workspace_id: str,
        paper_id: str,
    ) -> dict[str, Any]:
        """
        Read back the persisted paper card for a paper.

        Returns the full card dict if one exists, or {exists: false}.
        Useful when the agent wants to check whether a card has already
        been submitted before producing one, or to read the
        workspace-relevance framing of a sibling paper.
        """
        if (err := enforce(workspace_id, write=False)) is not None:
            return err

        ws_settings = state.settings.for_workspace(workspace_id)
        card = load_card(ws_settings.data_dir, paper_id)
        if card is None:
            return {"exists": False, "paper_id": paper_id}
        return {"exists": True, "paper_id": paper_id, "card": card.to_dict()}
