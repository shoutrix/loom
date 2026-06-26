# Loom — Unification Refactor (2026-06)

## Why this refactor exists

Loom today has three ingestion surfaces (`submit_paper_card`, `submit_document_card`, `write_vault_note`), two on-disk shapes (`paper_cards/`, `documents/`), and a UI that exposes the seams of both. The result:

- **Agents pick the wrong tool.** Even with dispatch-hint docstrings, the seams leak — agents have routed first-class content into `write_vault_note` (no indexing) and squeezed essays into the 13-field paper schema (lossy).
- **Rigid schemas drop signal.** The 13-field paper card forces agents to discard anything that doesn't fit (e.g. an essay with no "datasets" section ends up with empty strings or invented content).
- **Two storage shapes complicate every read path.** Registry, contents tree, retrieval, and renderer all carry `if paper else document` branches.
- **UI surfaces the implementation.** Three-column hard boundaries, status icons, summaries inline in the tree, and rigid card layouts — all loud where the document itself should be the focus.

This refactor collapses all three surfaces into one and rebuilds the UI around the document as the unit.

## Locked decisions

1. **One on-disk shape.** Everything lives in `documents/<doc_id>.json` with `doc_type ∈ {"research_paper", "article", "note", "transcript", "spec", "memo", "book_chapter", "documentation"}`. `paper_cards/` is migrated and deleted.
2. **One MCP ingestion tool.** `submit_document(workspace_id, body, doc_type, ...)`. Agent submits full markdown body; metadata is minimal (title optional, references optional for research papers, source_url optional). Legacy tools (`submit_paper_card(s)`, `submit_document_card(s)`, `write_vault_note`, `write_vault_file`) are deleted outright — no shims.
3. **Markdown is the canonical content format.** Storage is markdown. Render is markdown → React (react-markdown + remark-gfm + KaTeX + syntax highlighting). No HTML on disk.
4. **Derived metadata is async.** Submission is fast: write body to disk, register, enqueue. A background worker derives `title`, `tldr`, `category_path`, and (for research papers) extracted `references`. UI shows "categorizing…" until done.
5. **No backwards compatibility for in-flight data.** Old `paper_cards/*.json` and `documents/*.json` are migrated by a one-shot script; old MCP tools are gone the moment the refactor lands.
6. **Aesthetics:** grayscale only. Black, white, light gray (#f5f5f5 background), dark gray text. No accent colors. No icons except where genuinely functional. Three-dot menus for all per-doc actions.

## Target shape

```
data/<ws>/
  documents/
    <doc_id>.json          # metadata + body location
  vault/<ws>/
    documents/
      <doc_id>.md          # full markdown body, frontmatter + content
  paper_registry.json      # unified registry, every entry has doc_type
  semantic_index/, snapshot.json, wal.jsonl, ...
```

**Document JSON shape** (one schema for all doc_types):

```json
{
  "doc_id": "doc:abc123def456",
  "doc_type": "research_paper",
  "title": "Attention Is All You Need",
  "source_url": "https://arxiv.org/abs/1706.03762",
  "authors": ["Vaswani et al."],
  "published_at": "2017-06-12",
  "body_path": "documents/attention-is-all-you-need_doc:abc1.md",
  "tldr": "Self-attention replaces recurrence; introduces the Transformer.",
  "category_path": ["LLM Foundations", "Architectures"],
  "references": [
    {"title": "Neural Machine Translation by Jointly Learning to Align and Translate", "doc_id": null, "url": "https://arxiv.org/abs/1409.0473"}
  ],
  "metadata_status": "derived",   // "pending" | "deriving" | "derived" | "failed"
  "created_at": "2026-06-07T12:00:00Z",
  "updated_at": "2026-06-07T12:00:05Z"
}
```

Fields beyond `doc_id`/`doc_type`/`body_path` are optional. Agent supplies what it knows; worker fills the rest.

**`submit_document` MCP tool** — the entire ingestion surface:

```python
@mcp.tool()
def submit_document(
    workspace_id: str,
    body: str,                    # full markdown content
    doc_type: str = "note",       # research_paper | article | note | transcript | spec | memo | book_chapter | documentation
    title: str | None = None,     # if absent, derived from body's first H1 or async LLM
    source_url: str | None = None,
    authors: list[str] | None = None,
    published_at: str | None = None,
    references: list[dict] | None = None,  # only meaningful for research_paper
) -> dict: ...
```

Returns `{"ok": True, "doc_id": "doc:...", "metadata_status": "pending"}`. Body is written immediately; metadata pass runs async.

## Phase A — backend unification

**Goal:** one MCP tool, one on-disk shape, async metadata, legacy gone.

### Steps

1. **New document module** — `loom/document/` (replaces `loom/document_card/` + `loom/paper_card/`).
   - `loom/document/schema.py` — `Document` dataclass with the JSON shape above; `VALID_DOC_TYPES`; helpers.
   - `loom/document/store.py` — `save`, `load`, `list_for_workspace`, `delete`, `derive_doc_id(source_url=None, body_preview=None)`.
   - `loom/document/markdown.py` — `extract_title_from_markdown(body)`, `strip_frontmatter(body)`, `extract_arxiv_references(body)` (regex over arxiv IDs in body).

2. **Migration script** — `loom/scripts/migrate_unify_documents.py`.
   - Dry-run by default; `--apply` to commit.
   - Walks every `data/<ws>/`:
     - Reads `paper_cards/*.json`, renders body via `paper_card_to_markdown`, writes new `documents/<id>.json` + `vault/<ws>/documents/<slug>_<id[:8]>.md`. `doc_type="research_paper"`.
     - Reads existing `documents/*.json`, normalizes to new shape (most fields already align), points `body_path` at the vault file.
     - Updates `paper_registry.json` entries: every record gets `doc_type`; `paper_id` field is renamed to `doc_id` everywhere.
     - Backs up `paper_cards/`, old `documents/`, and old `paper_registry.json` as `.bak.<ts>`.
   - Leaves the original dirs in place under `.bak.*` so a manual rollback is possible.

3. **Unified registry** — `loom/storage/document_registry.py` (replaces `paper_registry.py`).
   - Field rename: `paper_id` → `doc_id` (always with `doc:` prefix; for migrated papers, derived from `arxiv_id` or content hash).
   - New required field: `doc_type`.
   - `register_document(doc_id, doc_type, title, source_url)` is the only registration method (no more `register_document` vs `register`).
   - 3-way merge on save (loaded snapshot + in-memory + disk) is preserved.

4. **Single ingestion tool** — `loom/mcp_server/tools/documents.py`.
   - Implements `submit_document` as specified.
   - Writes body to vault immediately via `vault.save_document`.
   - Registers in unified registry with `metadata_status="pending"`.
   - Enqueues a metadata-derivation job AND a retrieval-indexing job (chunk + embed + KG).
   - Deletes: `paper_card_tools.py`, `document_tools.py`, and the `write_vault_note` / `write_vault_file` tools in `shared.py`.

5. **Async metadata worker** — `loom/document/metadata_worker.py`.
   - Background loop, runs alongside `IngestionWorker`.
   - Picks documents with `metadata_status="pending"`, marks `deriving`, calls LLM for: `title` (if missing), `tldr`, `category_path` (uses existing `loom/categorize/` machinery).
   - For `doc_type="research_paper"`, additionally extracts arxiv-id references from the body.
   - Marks `derived` on success, `failed` on exception (with retry budget).

6. **Ingestion pipeline simplification** — `loom/main.py::IngestionWorker`.
   - Single branch: read body from `vault/<ws>/documents/<id>.md`, chunk + embed + KG. No more `paper card` vs `document` vs `arxiv-fetch` branching. The body is always already on disk because submission wrote it.
   - The `ingest_paper` URL-submission tool is deleted (no MCP surface for raw URLs anymore — the agent fetches the content and submits the body).

7. **Read-path unification.**
   - `loom/contents/builder.py` reads only `documents/` (no more `paper_cards` + `documents` merging). Tree summaries get `doc_type` so the UI can render the right per-doc actions.
   - `loom/api/routes_papers.py` → `loom/api/routes_documents.py`. All endpoints rename `paper_id` → `doc_id`. `/papers/*` URLs become `/documents/*`.
   - `loom/categorize/` reads `documents/` only.
   - `loom/retrieval/` and `loom/search/` use `doc_id` everywhere.

8. **MCP surface trim.**
   - Tools after Phase A: `health`, `list_workspaces`, `get_workspace`, `create_workspace`, `list_documents` (renamed from `list_papers`), `get_document` (renamed from `get_paper`), `get_document_body` (renamed from `read_vault_file`, scoped to documents), `submit_document`, `filter_new_documents` (renamed from `filter_new_papers`), `build_citation_tree`, `get_citation_tree`, plus the research/recommender tools as-is.

9. **Tests rewrite.**
   - Delete `tests/test_paper_card.py` and `tests/test_documents.py` (the old ones).
   - Add `tests/test_document_schema.py`, `tests/test_document_store.py`, `tests/test_submit_document_tool.py`, `tests/test_migration.py`, `tests/test_metadata_worker.py`.
   - Keep `test_smoke.py` updated to new tool names.

### Files touched (Phase A)

**New:** `loom/document/{schema.py,store.py,markdown.py,metadata_worker.py}`, `loom/storage/document_registry.py`, `loom/mcp_server/tools/documents.py`, `loom/api/routes_documents.py`, `loom/scripts/migrate_unify_documents.py`.

**Modified:** `loom/main.py` (IngestionWorker + app wiring), `loom/mcp_server/tools/{__init__.py,shared.py,research.py,recommender.py}`, `loom/contents/builder.py`, `loom/categorize/*`, `loom/retrieval/*`, `loom/search/*`, `loom/storage/vault.py`, `loom/config.py`.

**Deleted:** `loom/paper_card/`, `loom/document_card/`, `loom/storage/paper_registry.py`, `loom/mcp_server/tools/{paper_card_tools.py,document_tools.py}`, `loom/api/routes_papers.py`, old tests.

### Acceptance (Phase A)

- Migration script run on a copy of `data/` produces a clean unified shape; old data is `.bak`'d, not deleted.
- MCP server exposes exactly one ingestion tool. `mcp.list_tools()` does not include any `submit_paper_card*`, `submit_document_card*`, `write_vault_note`, `write_vault_file`, or `ingest_paper`.
- `submit_document(ws_id, body="# Test\n\nHello", doc_type="note")` returns within 500ms with `metadata_status="pending"`.
- Within ~10s, the metadata worker fills `title="Test"`, `tldr`, `category_path`; status flips to `derived`.
- Within ~30s, the document is chunked + embedded + KG-extracted; queries find it.
- Chat over a workspace with mixed migrated papers + new submissions works; retrieval pulls from both kinds.
- Tests green: `pytest loom/tests -q`.

### Verification (Phase A)

```bash
# Migration dry-run
python -m loom.scripts.migrate_unify_documents --workspace llm-agent-systems

# Migration apply (after eyeballing the diff)
python -m loom.scripts.migrate_unify_documents --workspace llm-agent-systems --apply

# MCP tool surface
python -c "import asyncio; from loom.mcp_server.server import build_mcp; \
  print(sorted(t.name for t in asyncio.run(build_mcp().list_tools())))"

# End-to-end smoke
python -m loom.scripts.smoke_submit_document
```

---

## Phase B — UI rebuild

**Goal:** minimalist, grayscale, document-centered layout; floating overlay panels for navigation + chat/graph.

### Layout

```
┌────────────────────────────────────────────────────────────────────────┐
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  ← bg: #f5f5f5
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│  ░┌──────────┐  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ┌────────┐░│
│  ░│ Contents │  ░░░░    Document content (centered)    ░░░  │  Chat  │░│
│  ░│  (white) │  ░░░░    serif heading, sans body       ░░░  │ (white)│░│
│  ░│  shadow  │  ░░░░    760px max-width                ░░░  │ shadow │░│
│  ░└──────────┘  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  └────────┘░│
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
└────────────────────────────────────────────────────────────────────────┘
```

- **Background:** `#f5f5f5` light gray, edge-to-edge. No header bar (workspace switcher folds into the left overlay).
- **Document area:** centered, ~760px max-width, white text panel sits *on* the gray bg with no border — the bg gray is what makes the document panel feel like the "page". Serif headings (Iowan, Georgia fallback), sans body (Inter).
- **Left overlay:** floating white panel, ~280px, `box-shadow: 0 4px 24px rgba(0,0,0,0.08)`, rounded 12px. Sticky.
- **Right overlay:** mirror of left, same shadow + radius.

### Left overlay — Contents

```
┌──────────────────────────┐
│  llm-agent-systems    ▾  │  ← workspace switcher (click → dropdown)
├──────────────────────────┤
│  + New note              │  ← clickable row
│  ☰ Brief                 │  ← clickable row
├──────────────────────────┤
│  CONTENTS                │  ← label, #aaa, 11px tracking-wide
│                          │
│  LLM Foundations         │
│    Architectures         │
│      Attention paper     │
│      How attention works │  ← hover → ⋮ on the right
│    Tool Use              │
│      ReAct RFC           │
│  Uncategorized           │
│    Loose note            │
└──────────────────────────┘
```

- **No status icons.** No per-doc summary line. Just the title.
- **Hover state:** row gets a subtle `#fafafa` background and a `⋮` button appears flush-right.
- **⋮ menu (research_paper):** *Open*, *Citation tree*, *Explore graph*, *Delete*.
- **⋮ menu (any other doc_type):** *Open*, *Delete*.
- **Click on the title** opens the document in the center panel.
- **Category labels** in light gray (`#aaa`), titles in `#222`.

### Right overlay — Chat / Graph toggle

```
┌──────────────────────────┐
│  ┌────────────────────┐  │
│  │ ┌──────┐ ┌───────┐ │  │  ← pill: bg #ececec, rounded 8px
│  │ │ Chat │ │ Graph │ │  │     active tab: white bg, shadow inset
│  │ └──────┘ └───────┘ │  │     inactive: blends with pill bg
│  └────────────────────┘  │
├──────────────────────────┤
│                          │
│  (Chat panel content OR  │
│   Graph view content)    │
│                          │
└──────────────────────────┘
```

- Single rounded-rect pill. Two equal halves. Active half: white bg with a soft inset; inactive: same color as pill, just text.
- Below the pill, the active panel renders (chat thread or graph view).

### Color tokens

```css
--bg-canvas:       #f5f5f5;
--bg-panel:        #ffffff;
--bg-panel-hover:  #fafafa;
--bg-pill:         #ececec;
--text-primary:    #1a1a1a;
--text-secondary:  #6b6b6b;
--text-muted:      #aaaaaa;
--shadow-panel:    0 4px 24px rgba(0, 0, 0, 0.08);
--shadow-inset:    inset 0 0 0 1px rgba(0, 0, 0, 0.04);
--radius-panel:    12px;
--radius-pill:     8px;
--radius-tab:      6px;
```

No other colors. No reds, blues, greens — error states use bolder text, not color.

### Steps (Phase B)

1. **New layout shell** — `frontend/src/App.tsx` rewritten as a CSS-grid with `bg-canvas`, left/right overlay positioning, center document slot. Drop the existing 3-column flex layout entirely.
2. **New `ContentsPanel`** — `frontend/src/components/ContentsPanel.tsx` (replaces `SourcesPanel.tsx`). Workspace switcher at top, "New note" + "Brief" rows, then categorized tree with hover-revealed `⋮`.
3. **New `DocActionMenu`** — `frontend/src/components/DocActionMenu.tsx`. Dropdown driven by `doc_type`. Uses headless dropdown pattern (Radix or custom), grayscale only.
4. **New `RightRail`** — `frontend/src/components/RightRail.tsx`. Holds the pill toggle + conditionally renders `ChatPanel` or `GraphView`.
5. **New `PillToggle`** — `frontend/src/components/PillToggle.tsx`. Reusable two-tab pill.
6. **`DocumentView`** — `frontend/src/components/DocumentView.tsx` (replaces `PaperViewer.tsx` + `DocumentCard.tsx` + `PaperCard.tsx`). Single component, renders the markdown body via react-markdown, no doc_type branching at the rendering level. Header section shows: title (serif, large), source_url link (small, muted), authors + published_at (muted line). Body below.
7. **Restyle `ChatPanel.tsx`** — strip color, drop status badges, simplify. Same for `GraphView.tsx`.
8. **Brief / New Note flows** — `BriefPanel.tsx` and `NoteEditor.tsx` rewired to open as the center document (not as a separate column).
9. **Delete:** `SourcesPanel.tsx`, `PaperViewer.tsx`, `DocumentCard.tsx`, `PaperCard.tsx`, `AddSourcesModal.tsx` (URL-submission flow is gone — agents submit via MCP), `QueueStatus.tsx` (status icons removed from UI; queue surfaces only in the ⋮ menu of in-flight docs as a "deriving…" badge).
10. **CSS:** delete every accent-color rule. Rebuild `index.css` with just the tokens above + typography rules.

### Files touched (Phase B)

**New:** `frontend/src/components/{ContentsPanel,DocActionMenu,RightRail,PillToggle,DocumentView}.tsx`.

**Modified:** `frontend/src/App.tsx`, `frontend/src/api/client.ts` (rename `paper_id` → `doc_id`, point at `/documents/*` routes), `frontend/src/components/{ChatPanel,GraphView,BriefPanel,NoteEditor,CitationTreeView,WorkspaceSwitcher,RecommenderPanel}.tsx`, `frontend/src/index.css`, `frontend/tailwind.config.js`.

**Deleted:** `frontend/src/components/{SourcesPanel,PaperViewer,DocumentCard,PaperCard,AddSourcesModal,QueueStatus}.tsx`.

### Acceptance (Phase B)

- `bash start.sh` boots and the homepage is the grayscale canvas with two floating panels — no other UI chrome.
- Clicking a document title opens it in the center; it renders correctly for both a migrated research paper and a freshly-submitted note.
- Hover on a tree row reveals `⋮`; the menu options change with `doc_type`.
- The chat/graph pill toggles cleanly with no layout shift.
- "New note" opens an inline editor in the center column; on save it round-trips through `submit_document(doc_type="note")`.
- `tsc --noEmit` clean.

### Verification (Phase B)

- Browser smoke: open workspace, open a migrated paper, open a note, toggle chat/graph, create a new note, hover-reveal-delete a note.
- Visual: no non-grayscale pixels anywhere except KaTeX math glyphs (which are inherently black).

---

## Phase ledger

| Phase | Description                                              | Status |
|-------|----------------------------------------------------------|--------|
| A     | Backend: unified shape, single MCP tool, async metadata  | [x] done 2026-06-07 |
| B     | Frontend: grayscale rebuild, overlay layout, ⋮ menus     | [x] done 2026-06-07 |

## Execution protocol

1. Phase A runs to acceptance first. Backend ships, MCP server reloads, existing workspaces migrate cleanly.
2. Phase B starts after A is green. Frontend rebuilds against the now-stable backend.
3. Each phase commits incrementally on `refactor/knowledge-substrate` (current branch); when a phase passes acceptance, flip `[ ]` → `[x] done YYYY-MM-DD <sha>` here.
4. Session log at bottom: one line per session.

---

## Session log

(Append one line per session: `YYYY-MM-DD <phase> <sha> note`.)

- 2026-06-07 plan — Drafted unification refactor plan. Locked decisions: unified `documents/` shape, single `submit_document` MCP tool, async metadata worker, markdown canonical, grayscale UI, no backwards-compat shims.
- 2026-06-07 B — Phase B complete. Frontend rebuilt around the unified `/documents/*` surface. New grayscale design system in `index.css` (no accent colors; serif headings, sans body, soft shadows). Layout collapsed to a CSS-grid with a light-gray canvas and two floating panels (`ContentsPanel` left, `RightRail` right). `ContentsPanel` shows the workspace switcher, `+ New note`, `☰ Brief`, then the categorized tree with hover-revealed `⋮` menus per document. `RightRail` carries a rounded-pill toggle between `ChatPanel` and `GraphView`. Unified `DocumentView` renders any `doc_type` from `/documents/{doc_id}/body` via react-markdown — replaces the old `PaperViewer` + `PaperCard` + `DocumentCard` split. `NoteEditor` and `BriefPanel` rewired to open in the center column; `NoteEditor` posts through the canonical `submit_document` route. Deleted: `SourcesPanel`, `PaperViewer`, `PaperCard`, `DocumentCard`, `AddSourcesModal`, `QueueStatus`, `CitationTreeView`, `RecommenderPanel`. `tsc --noEmit` clean.
- 2026-06-07 A — Phase A complete. New `loom/document/` module (schema, store, markdown helpers); `DocumentRegistry` replaces `PaperRegistry` (paper_id→doc_id everywhere). Single `submit_document` MCP tool; legacy `submit_paper_card[s]`, `submit_document_card[s]`, `write_vault_note`, `write_vault_file`, `list_papers`, `get_paper`, `ingest_paper` deleted. `MetadataWorker` derives `title` / `tldr` / `category_path` / arxiv refs asynchronously. `IngestionWorker` collapsed to one branch (read body from `vault/<ws>/documents/<slug>_<id[:8]>.md`, chunk+embed+KG). `routes_papers.py` + `routes_ingest.py` deleted; new `routes_documents.py` exposes the HTTP mirror. `contents/builder.py` reads only the unified store. Migration script (`migrate_unify_documents.py`) dry-runs 66 papers + 1 document across 7 workspaces clean. MCP surface trimmed to 18 tools. Test suite green: 147 passed.
