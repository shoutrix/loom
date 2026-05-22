"""
Paper-submission MCP tools — the entire MCP "write side" for papers.

The MCP boundary is intentionally narrow: any subscriber can deposit
papers into a workspace in one of two shapes — a URL (loom will analyze
it) or a pre-built structured review card (loom will skip its own
analysis and just persist the agent's review). What loom does with the
deposit — chunking, embedding, knowledge-graph extraction — is loom's
concern, not the subscriber's.

Tools exported here:

  submit_paper(workspace_id, url)
      Deposit a single paper URL. Loom queues it for background
      ingestion. Returns immediately.

  submit_papers(workspace_id, urls[])
      Batch version (capped at MAX_BATCH_SIZE = 10).

  submit_paper_card(workspace_id, card)
      Deposit a paper with the agent's pre-built structured review. The
      card must contain source_url (or arxiv_id / doi). Loom queues the
      paper for ingestion AND persists the card immediately — so when
      the user opens the paper in the UI, loom does NOT need to re-run
      its Gemini card-extraction pass.

  submit_paper_cards(workspace_id, cards[])
      Batch version (capped at MAX_BATCH_SIZE).

  get_paper_card(workspace_id, paper_id)
      Read-back tool. Returns the persisted card if one exists.

Processing model:
  All four submit tools are FIRE-AND-FORGET. They write the paper to
  the workspace's paper_registry with status='queued' (and the card to
  paper_cards/ when supplied), then return in well under a second.
  Loom's FastAPI-side IngestionWorker scans every workspace's registry
  on startup and every 60 seconds, picks up queued papers, and processes
  them in the background using loom's configured server-side LLM (Gemini
  by default — see LOOM_LLM_PROVIDER). The agent's session does not
  need to remain alive for ingestion to complete.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from loom.mcp_server.state import MCPState
from loom.mcp_server.workspace import MCPWorkspaceLoader
from loom.paper_card import PaperCard, save_card, load_card
from loom.paper_card.extractor import (
    _coerce_datasets,
    _coerce_related_work,
    _coerce_str_list,
)
from loom.permissions import enforce
from loom.storage.paper_registry import PaperRegistry


# Batch size cap for the *_papers tools. Above this the agent gets a
# clear error message rather than silent truncation.
MAX_BATCH_SIZE = 10


# ----- helpers ---------------------------------------------------------------


def _registry_for(workspace_data_dir: Path) -> PaperRegistry:
    """Open / create the paper_registry.json for a workspace."""
    workspace_data_dir.mkdir(parents=True, exist_ok=True)
    return PaperRegistry(workspace_data_dir / "paper_registry.json")


def _find_existing(registry: PaperRegistry, url: str) -> Any:
    """Return the existing PaperRecord matching this URL/arxiv_id/doi, or None.

    Match priority: source_url → arxiv_id (extracted from URL) → doi.
    This is the dedup primitive — callers use it to detect re-submissions
    BEFORE register_and_queue, which would otherwise overwrite the status
    of an already-ingested paper.
    """
    import re

    url = (url or "").strip()
    if not url:
        return None

    # Extract identifiers from the URL.
    arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url)
    arxiv_id = arxiv_match.group(1) if arxiv_match else None
    if not arxiv_id and re.fullmatch(r"\d{4}\.\d{4,5}", url):
        arxiv_id = url

    doi_match = re.search(r"doi\.org/(.+)$", url)
    doi = doi_match.group(1) if doi_match else None
    if not doi and url.startswith("10."):
        doi = url

    for rec in registry.get_all():
        if arxiv_id and getattr(rec, "arxiv_id", "") == arxiv_id:
            return rec
        if doi and getattr(rec, "doi", "") == doi:
            return rec
    return None


def _existing_response(rec, action: str, *, message: str | None = None) -> dict[str, Any]:
    """Build a uniform 'already exists' response for submit_* tools."""
    return {
        "ok": True,
        "paper_id": rec.paper_id,
        "status": rec.status,
        "action": action,
        "title": getattr(rec, "title", "") or "",
        "ingested_at": getattr(rec, "ingested_at", "") or "",
        "message": message or (
            f"Paper is already in this workspace with status={rec.status!r}. "
            f"No re-submission needed."
        ),
    }


def _derive_url(card: dict[str, Any]) -> str | None:
    """Best-effort URL extraction from a submitted card.

    Order of preference: source_url > arxiv_id > doi.
    """
    url = card.get("source_url")
    if isinstance(url, str) and url.strip():
        return url.strip()
    arxiv_id = card.get("arxiv_id")
    if isinstance(arxiv_id, str) and arxiv_id.strip():
        return f"https://arxiv.org/abs/{arxiv_id.strip()}"
    doi = card.get("doi")
    if isinstance(doi, str) and doi.strip():
        return f"https://doi.org/{doi.strip()}"
    return None


def _coerce_card_input(
    paper_id: str,
    card_in: dict[str, Any],
    *,
    registry_header: dict[str, Any] | None = None,
) -> PaperCard:
    """Fold an agent-submitted card dict into a PaperCard dataclass.

    Header fields fall back to ``registry_header`` so the agent doesn't
    have to re-supply bibliographic data loom already has. List fields
    are tolerant of common shape variations (strings-where-objects-
    expected, missing fields, etc.).
    """
    import datetime

    rh = registry_header or {}

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

    title = _str(card_in.get("title")) or _str(rh.get("title"))
    authors = _str_list(card_in.get("authors")) or _str_list(rh.get("authors"))
    venue = _str(card_in.get("venue")) or _str(rh.get("venue"))
    year = _maybe_int(card_in.get("year"))
    if year is None and isinstance(rh.get("year"), int):
        year = rh["year"]
    source_url = _str(card_in.get("source_url")) or _str(rh.get("source_url"))
    arxiv_id = _str(card_in.get("arxiv_id")) or _str(rh.get("arxiv_id"))
    doi = _str(card_in.get("doi")) or _str(rh.get("doi"))

    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    return PaperCard(
        version=1,
        generated_at=now,
        model="agent-submitted",
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


def _registry_header(registry: PaperRegistry, paper_id: str) -> dict[str, Any]:
    """Build the header dict from the registry record (best-effort)."""
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


# ----- registration ----------------------------------------------------------


def register(mcp: FastMCP, state: MCPState, loader: MCPWorkspaceLoader) -> None:

    @mcp.tool()
    async def submit_paper(
        workspace_id: str,
        url: str,
    ) -> dict[str, Any]:
        """
        Deposit one paper URL into a workspace.

        Loom queues the paper for background processing. The processing
        details (parsing, indexing, knowledge-graph extraction) are
        loom-internal — you, the calling agent, don't need to know
        about them. This call returns in well under a second.

        If you've already read the paper and produced a structured
        review, prefer `submit_paper_card` instead — that lets loom
        skip its own card-extraction step entirely.

        Args:
            workspace_id: target workspace (must be writable for this
                subscriber).
            url: an identifier loom can resolve — typically an arXiv
                URL ("https://arxiv.org/abs/2401.12345"), a bare arXiv
                ID ("2401.12345"), a DOI ("10.1234/foo"), an S2 id
                ("s2:<id>"), or any web URL loom should fetch.

        Returns:
            {ok, paper_id, workspace_id, status: "queued", message}
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        if not isinstance(url, str) or not url.strip():
            return {"ok": False, "error": "url must be a non-empty string"}

        ws_settings = state.settings.for_workspace(workspace_id)
        registry = _registry_for(ws_settings.data_dir)

        # Dedup: if a record for this URL already exists, surface its
        # current status rather than re-queueing. Skip-noop for ingested
        # papers; idempotent for queued/ingesting; allow retry for failed.
        existing = _find_existing(registry, url.strip())
        if existing and existing.status in ("ingested", "ingesting", "queued"):
            return _existing_response(existing, action="skipped")

        paper_id = registry.register_and_queue(url.strip())
        registry.save()

        return {
            "ok": True,
            "paper_id": paper_id,
            "workspace_id": workspace_id,
            "status": "queued",
            "action": "queued",
            "message": (
                "Paper queued. Loom's background worker will pick it up "
                "within 60 seconds and ingest it (~30-60s of processing). "
                "If you have a structured review for this paper, you can "
                "still submit it via submit_paper_card now — loom will "
                "match it on paper_id."
            ),
        }

    @mcp.tool()
    async def submit_papers(
        workspace_id: str,
        urls: list[str],
    ) -> dict[str, Any]:
        """
        Deposit multiple paper URLs at once. Capped at 10 per call.

        Returns:
            {ok, total, submissions: [{url, paper_id, status} | {url, error}]}
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        if not isinstance(urls, list):
            return {"ok": False, "error": "urls must be a list of strings"}
        if len(urls) > MAX_BATCH_SIZE:
            return {
                "ok": False,
                "error": (
                    f"max batch size is {MAX_BATCH_SIZE}; got {len(urls)}. "
                    f"Split into smaller batches."
                ),
            }

        ws_settings = state.settings.for_workspace(workspace_id)
        registry = _registry_for(ws_settings.data_dir)

        submissions: list[dict[str, Any]] = []
        skipped = 0
        queued = 0
        for raw in urls:
            if not isinstance(raw, str) or not raw.strip():
                submissions.append({"url": raw, "error": "empty or non-string url"})
                continue
            try:
                existing = _find_existing(registry, raw.strip())
                if existing and existing.status in ("ingested", "ingesting", "queued"):
                    submissions.append({
                        "url": raw.strip(),
                        "paper_id": existing.paper_id,
                        "status": existing.status,
                        "action": "skipped",
                    })
                    skipped += 1
                    continue
                pid = registry.register_and_queue(raw.strip())
                submissions.append({
                    "url": raw.strip(),
                    "paper_id": pid,
                    "status": "queued",
                    "action": "queued",
                })
                queued += 1
            except Exception as e:
                submissions.append({"url": raw, "error": f"{type(e).__name__}: {e}"})
        registry.save()

        return {
            "ok": True,
            "total": len(submissions),
            "queued": queued,
            "skipped": skipped,
            "workspace_id": workspace_id,
            "submissions": submissions,
        }

    @mcp.tool()
    async def submit_paper_card(
        workspace_id: str,
        card: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Deposit a paper together with a pre-built structured review.

        Use this when you've already read and analyzed the paper in your
        own context (e.g. via WebFetch). Loom persists the card as-is —
        bypassing its own Gemini-based card extractor that would
        otherwise run lazily on first UI view.

        Loom still queues the paper for background ingestion (chunking,
        embedding, knowledge-graph extraction). Those steps are loom-
        internal and don't affect you — they happen on loom's pace
        using loom's configured LLM.

        REQUIRED CARD SCHEMA (the 13 review fields)
        ===========================================

        The card MUST identify the paper via one of these:
          - source_url (preferred, e.g. "https://arxiv.org/abs/2603.13686")
          - arxiv_id   (e.g. "2603.13686")
          - doi        (e.g. "10.1234/foo")

        All 13 review fields below should be populated when the paper
        supports them. The UI hides empty fields silently — leaving
        them blank degrades the user's view.

        {
          "source_url": "<url; required if no arxiv_id/doi>",
          "arxiv_id": "<optional>",
          "doi": "<optional>",
          "title": "<optional; if omitted, falls back to registry>",
          "authors": ["<optional>"],
          "venue": "<optional>",
          "year": <optional, int>,

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
              for THIS workspace specifically.>",
          "open_questions": ["<bullet>", ...]
        }

        Extra fields beyond the schema are silently ignored.

        Returns:
            {ok, paper_id, workspace_id, status, card_saved, fields_filled,
             total_fields, message}
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        if not isinstance(card, dict):
            return {
                "ok": False,
                "error": f"`card` must be a JSON object, got {type(card).__name__}",
            }

        url = _derive_url(card)
        if not url:
            return {
                "ok": False,
                "error": "card must contain `source_url`, `arxiv_id`, or `doi`",
            }

        ws_settings = state.settings.for_workspace(workspace_id)
        registry = _registry_for(ws_settings.data_dir)

        # Dedup: if the paper is already in the registry, ALWAYS overwrite
        # the card (the agent's new analysis is presumably newer/better)
        # but do NOT re-queue an already-ingested paper.
        existing = _find_existing(registry, url)
        if existing:
            paper_id = existing.paper_id
            action_for_paper = (
                "skipped" if existing.status in ("ingested", "ingesting", "queued")
                else "queued"
            )
            if action_for_paper == "queued":
                registry.register_and_queue(url)
            existing_status_for_card = existing.status
        else:
            paper_id = registry.register_and_queue(url)
            action_for_paper = "queued"
            existing_status_for_card = "queued"
        registry.save()

        header = _registry_header(registry, paper_id)
        built = _coerce_card_input(paper_id, card, registry_header=header)
        save_card(ws_settings.data_dir, built)

        fields_filled = sum([
            bool(built.tldr), bool(built.problem), bool(built.approach),
            len(built.contributions) > 0, len(built.datasets) > 0,
            bool(built.setup), len(built.results) > 0, bool(built.conclusion),
            len(built.strengths) > 0, len(built.limitations) > 0,
            len(built.related_work) > 0, bool(built.workspace_relevance),
            len(built.open_questions) > 0,
        ])

        return {
            "ok": True,
            "paper_id": paper_id,
            "workspace_id": workspace_id,
            "status": existing_status_for_card,
            "action": (
                "card_updated" if existing and action_for_paper == "skipped"
                else action_for_paper
            ),
            "card_saved": True,
            "fields_filled": fields_filled,
            "total_fields": 13,
            "message": (
                f"Card persisted ({fields_filled}/13 fields). "
                + ("Paper was already in the workspace — only the card was updated; ingestion not re-triggered."
                   if existing and action_for_paper == "skipped"
                   else "Paper queued for background ingestion.")
            ),
        }

    @mcp.tool()
    async def submit_paper_cards(
        workspace_id: str,
        cards: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Deposit multiple paper cards at once. Capped at 10 per call.

        Each item is processed independently — one bad card doesn't
        abort the rest. Returns per-item outcomes.

        See `submit_paper_card` for the card schema.

        Returns:
            {ok, total, succeeded, failed, submissions: [...]}
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        if not isinstance(cards, list):
            return {"ok": False, "error": "cards must be a list of card objects"}
        if len(cards) > MAX_BATCH_SIZE:
            return {
                "ok": False,
                "error": (
                    f"max batch size is {MAX_BATCH_SIZE}; got {len(cards)}. "
                    f"Split into smaller batches."
                ),
            }

        ws_settings = state.settings.for_workspace(workspace_id)
        registry = _registry_for(ws_settings.data_dir)

        submissions: list[dict[str, Any]] = []
        succeeded = 0
        for i, card in enumerate(cards):
            if not isinstance(card, dict):
                submissions.append({
                    "index": i,
                    "error": f"item {i} must be a JSON object, got {type(card).__name__}",
                })
                continue
            url = _derive_url(card)
            if not url:
                submissions.append({
                    "index": i,
                    "error": "card must contain source_url, arxiv_id, or doi",
                })
                continue
            try:
                existing = _find_existing(registry, url)
                if existing:
                    paper_id = existing.paper_id
                    item_action = (
                        "card_updated"
                        if existing.status in ("ingested", "ingesting", "queued")
                        else "queued"
                    )
                    if item_action == "queued":
                        registry.register_and_queue(url)
                    item_status = existing.status
                else:
                    paper_id = registry.register_and_queue(url)
                    item_action = "queued"
                    item_status = "queued"

                header = _registry_header(registry, paper_id)
                built = _coerce_card_input(paper_id, card, registry_header=header)
                save_card(ws_settings.data_dir, built)
                submissions.append({
                    "index": i,
                    "url": url,
                    "paper_id": paper_id,
                    "status": item_status,
                    "action": item_action,
                    "card_saved": True,
                })
                succeeded += 1
            except Exception as e:
                submissions.append({
                    "index": i,
                    "url": url,
                    "error": f"{type(e).__name__}: {e}",
                })
        registry.save()

        return {
            "ok": True,
            "total": len(cards),
            "succeeded": succeeded,
            "failed": len(cards) - succeeded,
            "workspace_id": workspace_id,
            "submissions": submissions,
        }

    @mcp.tool()
    async def get_paper_card(
        workspace_id: str,
        paper_id: str,
    ) -> dict[str, Any]:
        """
        Read back the persisted paper card for a paper.

        Returns the full card dict if one exists, or {exists: false}.
        Useful to check whether a card has already been submitted before
        producing a new one.
        """
        if (err := enforce(workspace_id, write=False)) is not None:
            return err
        ws_settings = state.settings.for_workspace(workspace_id)
        card = load_card(ws_settings.data_dir, paper_id)
        if card is None:
            return {"exists": False, "paper_id": paper_id}
        return {"exists": True, "paper_id": paper_id, "card": card.to_dict()}

    @mcp.tool()
    async def list_workspace_papers(
        workspace_id: str,
        status_filter: str = "",
    ) -> dict[str, Any]:
        """
        Enumerate every paper in a workspace's registry, with identifiers.

        Returns a compact list — paper_id, title, status, arxiv_id, doi,
        source_url — that the calling agent can scan against its own
        candidate set to avoid analyzing duplicates BEFORE producing a
        card.

        Args:
            workspace_id: target workspace.
            status_filter: optional — restrict to one status.
                Valid values: "shortlisted", "queued", "ingesting",
                "ingested", "failed". Empty string returns every paper
                regardless of status.

        Returns:
            {
              "workspace_id": ...,
              "total": <int>,
              "papers": [
                {
                  "paper_id": ...,
                  "title": ...,
                  "status": ...,
                  "arxiv_id": ...,
                  "doi": ...,
                  "source_url": ...,
                  "has_card": <bool — whether a card has been persisted>
                },
                ...
              ]
            }

        Use this once at the start of a bulk-analyze workflow, then dedup
        your candidate list against the returned URLs/IDs in your own
        context.
        """
        if (err := enforce(workspace_id, write=False)) is not None:
            return err

        ws_settings = state.settings.for_workspace(workspace_id)
        registry = _registry_for(ws_settings.data_dir)
        all_recs = registry.get_all()
        if status_filter:
            all_recs = [r for r in all_recs if r.status == status_filter]

        cards_dir = ws_settings.data_dir / "paper_cards"

        def _src(rec) -> str:
            if rec.arxiv_id:
                return f"https://arxiv.org/abs/{rec.arxiv_id}"
            if rec.doi:
                return f"https://doi.org/{rec.doi}"
            return ""

        papers = [
            {
                "paper_id": rec.paper_id,
                "title": rec.title or "",
                "status": rec.status,
                "arxiv_id": rec.arxiv_id or "",
                "doi": rec.doi or "",
                "source_url": _src(rec),
                "has_card": (cards_dir / f"{rec.paper_id}.json").exists()
                if cards_dir.exists() else False,
            }
            for rec in all_recs
        ]
        return {
            "workspace_id": workspace_id,
            "total": len(papers),
            "papers": papers,
        }

    @mcp.tool()
    async def build_citation_tree(
        workspace_id: str,
        paper_id_or_url: str,
        depth: int = 3,
        per_hop_cap: int = 80,
        max_nodes: int = 500,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        """
        Build (or rebuild) the citation tree for a paper. BLOCKING — runs
        the full pipeline end-to-end and returns the result.

        This is the primary tool for the "find everything related to this
        paper" workflow. After it returns, the agent has the full ranked
        candidate list to analyze + submit downstream.

        Pipeline (loom-internal — agent doesn't need to know):
          1. Bounded BFS on S2 references + citations (depth, per_hop_cap).
          2. Per-node signals (citation count, methodology ratio, etc.).
          3. PageRank + time-balanced PageRank.
          4. Convergence path counting.
          5. Composite scoring + 5-tier classification.
          6. (Optional) LLM pass using loom's server-side model with the
             full available context window — fills Gemini's 1M-token
             window with all candidate abstracts.

        Args:
            workspace_id: target workspace (where to persist the tree).
            paper_id_or_url: the paper to root the tree at. URL, arxiv
                ID, DOI, or s2:<id> all work. The paper does NOT have to
                already be in the workspace.
            depth: hops in each direction. Default 3.
            per_hop_cap: max papers consumed into the next frontier per
                hop. Default 80.
            max_nodes: hard ceiling on subgraph size. Default 500.
            use_llm: whether to run the LLM classification pass. Costs
                ~$2-5 in Gemini tokens. Strongly recommended (default).

        Wall-clock: 30-180 seconds depending on depth and S2 latency.

        Returns:
            {
              "ok": True,
              "target": {"paper_id", "title", "year", ...},
              "tiers": {
                "origin":      [paper_ids…],
                "landmark":    [...],
                "target":      [target_paper_id],
                "convergence": [...],
                "frontier":    [...]
              },
              "all_candidates": [
                # Full ranked list — what the agent uses for downstream
                # analysis. Score-sorted desc.
                {"paper_id", "title", "year", "score_influence",
                 "tier", "citation_count", "convergence_count", ...},
                ...
              ],
              "stats": {wall_time_total_seconds, total_nodes, ...},
              "path": "<on-disk location of the persisted JSON>"
            }
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        if not isinstance(paper_id_or_url, str) or not paper_id_or_url.strip():
            return {"ok": False, "error": "paper_id_or_url must be a non-empty string"}

        from loom.citation_tree import BuildParams, build_citation_tree, tree_path
        from loom.config import get_settings
        from loom.llm import make_llm_provider
        from loom.tools.paper_search.sources import SemanticScholarClient

        settings = get_settings()
        ws_settings = settings.for_workspace(workspace_id)
        ws_settings.ensure_dirs()

        s2 = SemanticScholarClient(api_key=settings.semantic_scholar_api_key or None)
        llm = make_llm_provider(settings, workspace_id=workspace_id) if use_llm else None

        try:
            tree = await asyncio.to_thread(
                build_citation_tree,
                paper_id_or_url.strip(),
                ws_settings.data_dir,
                s2_client=s2,
                llm=llm,
                params=BuildParams(
                    depth=depth,
                    per_hop_cap=per_hop_cap,
                    max_nodes=max_nodes,
                    use_llm_tiebreak=use_llm,
                ),
            )
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

        # Build a compact ranked candidate list, sorted by score_influence desc.
        candidates: list[dict[str, Any]] = []
        if tree.subgraph is not None:
            for pid, node in tree.subgraph.nodes.items():
                sig = tree.signals.get(pid)
                if sig is None:
                    continue
                src = ""
                if node.arxiv_id:
                    src = f"https://arxiv.org/abs/{node.arxiv_id}"
                elif node.doi:
                    src = f"https://doi.org/{node.doi}"
                elif node.url:
                    src = node.url
                candidates.append({
                    "paper_id": pid,
                    "title": node.title,
                    "year": node.year,
                    "venue": node.venue,
                    "source_url": src,
                    "arxiv_id": node.arxiv_id,
                    "doi": node.doi,
                    "citation_count": node.citation_count,
                    "influential_citation_count": node.influential_citation_count,
                    "score_influence": sig.score_influence,
                    "score_origin": sig.score_origin,
                    "score_frontier": sig.score_frontier,
                    "local_pagerank": sig.local_pagerank,
                    "convergence_count": sig.convergence_count,
                    "tier": sig.tier,
                })
        candidates.sort(key=lambda c: c["score_influence"], reverse=True)

        return {
            "ok": True,
            "target": tree.target,
            "tiers": tree.tiers,
            "all_candidates": candidates,
            "stats": tree.stats,
            "path": str(tree_path(ws_settings.data_dir, tree.target.get("paper_id", ""))),
        }

    @mcp.tool()
    async def get_citation_tree(
        workspace_id: str,
        paper_id: str,
    ) -> dict[str, Any]:
        """
        Read back the persisted citation tree for a paper, if one exists.

        Returns the same shape as `build_citation_tree` — target, tiers,
        all_candidates, stats — but loaded from disk in <10ms with no
        S2 or LLM calls.

        Use this when you want to check whether a tree has already been
        built (e.g. before kicking off a new `build_citation_tree`).
        """
        if (err := enforce(workspace_id, write=False)) is not None:
            return err

        from loom.citation_tree import load_tree, resolve_target_id

        ws_settings = state.settings.for_workspace(workspace_id)
        resolved = resolve_target_id(paper_id)
        tree = load_tree(ws_settings.data_dir, resolved)
        if tree is None:
            return {"exists": False, "paper_id": resolved}

        candidates: list[dict[str, Any]] = []
        if tree.subgraph is not None:
            for pid, node in tree.subgraph.nodes.items():
                sig = tree.signals.get(pid)
                if sig is None:
                    continue
                src = ""
                if node.arxiv_id:
                    src = f"https://arxiv.org/abs/{node.arxiv_id}"
                elif node.doi:
                    src = f"https://doi.org/{node.doi}"
                elif node.url:
                    src = node.url
                candidates.append({
                    "paper_id": pid,
                    "title": node.title,
                    "year": node.year,
                    "source_url": src,
                    "score_influence": sig.score_influence,
                    "convergence_count": sig.convergence_count,
                    "tier": sig.tier,
                })
        candidates.sort(key=lambda c: c["score_influence"], reverse=True)

        return {
            "exists": True,
            "target": tree.target,
            "tiers": tree.tiers,
            "all_candidates": candidates,
            "stats": tree.stats,
            "generated_at": tree.generated_at,
        }
