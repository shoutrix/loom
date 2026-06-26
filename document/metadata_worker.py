"""
Async derivation of `title` / `tldr` / `category_path` / `references`.

Submission is fast: `submit_document` writes the body + JSON descriptor
synchronously with `metadata_status="pending"` and returns. This worker
loops every WORKER_INTERVAL seconds and fills the derived fields for
all pending documents across all workspaces.

For research_paper documents, the worker also extracts arxiv IDs from
the body and merges them into the descriptor's `references` list (any
agent-supplied entries with the same arxiv_id are preserved).

The worker is best-effort: a failure on one document is logged via
`metadata_status="failed"` and `metadata_error` and the rest continue.
A subsequent re-submit resets status to `pending` and re-tries.
"""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from loom.config import Settings
from loom.contents.fitter import fit_paper_into_existing
from loom.document import (
    Reference,
    extract_arxiv_references,
    extract_title_from_markdown,
    list_documents,
    load_document,
    save_document,
    strip_frontmatter,
)
from loom.document.schema import Document

if TYPE_CHECKING:
    from loom.llm.base import LLMProvider

WORKER_INTERVAL = 30  # seconds between scans
MAX_BODY_FOR_TLDR = 8000  # chars; flash model handles ~32k tokens easily


_TLDR_PROMPT = (
    "Summarize the following document in 1 to 2 sentences. Be concrete: "
    "name the thing the document is about and the main claim or purpose. "
    "Avoid generic phrases like 'this document discusses' or 'this paper "
    "explores'. Return ONLY the summary text — no preamble, no quotes, no "
    "markdown.\n\nDocument:\n\n{body}"
)


class MetadataWorker:
    """Background derivation of `title` / `tldr` / `category_path` / refs."""

    def __init__(
        self,
        settings: Settings,
        llm_factory,
    ) -> None:
        """
        Args:
            settings: loom settings (used to enumerate workspaces).
            llm_factory: callable(workspace_id) -> LLMProvider. Lets the
                worker route per-workspace LLM context the way the rest
                of loom does (workspace-scoped usage tracking).
        """
        self._settings = settings
        self._llm_factory = llm_factory
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ----- lifecycle ----------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="document-metadata-worker",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ----- main loop ----------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                traceback.print_exc()
            self._stop.wait(WORKER_INTERVAL)

    def _tick(self) -> None:
        for ws_id in self._discover_workspaces():
            try:
                self._process_workspace(ws_id)
            except Exception:
                traceback.print_exc()

    def _discover_workspaces(self) -> list[str]:
        root = self._settings.data_dir
        if not root.exists():
            return []
        out: list[str] = []
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "workspace.json").exists():
                out.append(child.name)
        return out

    # ----- per-workspace processing ------------------------------------

    def _process_workspace(self, workspace_id: str) -> None:
        ws_settings = self._settings.for_workspace(workspace_id)
        docs = list_documents(ws_settings.data_dir)
        pending = [d for d in docs if d.metadata_status == "pending"]
        if not pending:
            return

        existing_paths = self._collect_existing_paths(docs)
        llm = None  # lazy: only build a provider when we actually need one

        for doc in pending:
            self._derive_one(doc, ws_settings, existing_paths, llm_getter=lambda: self._lazy_llm(workspace_id))

    def _lazy_llm(self, workspace_id: str):
        # Reuse the same provider per workspace per tick to amortize setup.
        if not hasattr(self, "_llm_cache"):
            self._llm_cache: dict[str, "LLMProvider"] = {}
        if workspace_id not in self._llm_cache:
            self._llm_cache[workspace_id] = self._llm_factory(workspace_id)
        return self._llm_cache[workspace_id]

    def _derive_one(
        self,
        doc: Document,
        ws_settings,
        existing_paths: list[list[str]],
        *,
        llm_getter,
    ) -> None:
        # Mark deriving immediately so concurrent ticks don't double-process.
        doc.metadata_status = "deriving"
        save_document(ws_settings.data_dir, doc)

        try:
            body_path = ws_settings.vault_dir / doc.body_path
            if not body_path.exists():
                doc.metadata_status = "failed"
                doc.metadata_error = f"body file missing: {doc.body_path}"
                save_document(ws_settings.data_dir, doc)
                return

            body = body_path.read_text(encoding="utf-8")
            stripped = strip_frontmatter(body)

            # 1. Title fallback (if agent didn't supply one).
            if not doc.title or doc.title.startswith("doc:"):
                title_from_body = extract_title_from_markdown(stripped)
                if title_from_body:
                    doc.title = title_from_body

            # 2. tldr derivation.
            if not doc.tldr:
                llm = llm_getter()
                try:
                    resp = llm.generate(
                        _TLDR_PROMPT.format(body=stripped[:MAX_BODY_FOR_TLDR]),
                        model="flash",
                        temperature=0.2,
                        max_output_tokens=256,
                    )
                    doc.tldr = (resp.text or "").strip()[:600]
                except Exception as e:
                    # Keep going — tldr is best-effort; categorization can
                    # still happen off the title.
                    doc.metadata_error = f"tldr derivation failed: {e}"

            # 3. category_path placement (only if agent didn't pre-set it).
            if not doc.category_path and existing_paths:
                llm = llm_getter()
                try:
                    chosen = fit_paper_into_existing(
                        llm,
                        title=doc.title,
                        abstract=doc.tldr,
                        existing_paths=existing_paths,
                        model="flash",
                    )
                    if chosen:
                        doc.category_path = chosen
                except Exception as e:
                    doc.metadata_error = f"categorization failed: {e}"

            # 4. References for research papers (arxiv extraction).
            if doc.doc_type == "research_paper":
                extracted = extract_arxiv_references(stripped)
                if extracted:
                    by_arxiv = {r.arxiv_id: r for r in doc.references if r.arxiv_id}
                    for axid in extracted:
                        if axid not in by_arxiv:
                            doc.references.append(Reference(
                                arxiv_id=axid,
                                url=f"https://arxiv.org/abs/{axid}",
                            ))

            doc.metadata_status = "derived"
            doc.metadata_error = ""
        except Exception as e:
            doc.metadata_status = "failed"
            doc.metadata_error = f"{type(e).__name__}: {e}"
        finally:
            save_document(ws_settings.data_dir, doc)

    @staticmethod
    def _collect_existing_paths(docs: list[Document]) -> list[list[str]]:
        """De-duplicated list of category_paths in the workspace, sorted by
        depth-then-name so the fitter prefers deeper matches."""
        seen: set[tuple[str, ...]] = set()
        out: list[list[str]] = []
        for d in docs:
            if d.category_path:
                t = tuple(d.category_path)
                if t not in seen:
                    seen.add(t)
                    out.append(list(d.category_path))
        out.sort(key=lambda p: (-len(p), tuple(p)))
        return out
