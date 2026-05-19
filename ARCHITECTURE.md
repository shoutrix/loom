# Loom architecture

## Substrate-first design

The knowledge base is the product. Loom UI, MCP server, recommender, and any
future agent are clients that subscribe to it with declared read/write scopes.

```
                              ┌────────────────────────────────────────┐
                              │           kbase (loom core)            │
                              │                                        │
                              │  Workspace (just a document container) │
                              │   ├── storage: snapshot, wal,          │
                              │   │   vault/, paper_registry,          │
                              │   │   keyword_index, semantic_index    │
                              │   ├── GraphStore                       │
                              │   ├── IngestionPipeline                │
                              │   └── Retriever interface              │
                              │        ├── FullContextRetriever        │
                              │        ├── GraphHybridRetriever        │
                              │        └── AdaptiveRetriever           │
                              │                                        │
                              │  Provider interfaces                   │
                              │   ├── LLMProvider                      │
                              │   │    ├── GeminiLLMProvider           │
                              │   │    ├── OpenRouterLLMProvider       │
                              │   │    └── MCPSamplingLLMProvider      │
                              │   └── EmbeddingProvider                │
                              │        └── GeminiEmbeddingProvider     │
                              │                                        │
                              │  permissions/SubscriberRegistry        │
                              └────────────────────────────────────────┘
                                 ▲           ▲              ▲
                  reads/writes   │           │              │
                  scoped by      │           │              │
                  SubscriberRegistry         │              │
                                 │           │              │
        ┌────────────────────────┴───┐   ┌───┴────────┐   ┌─┴──────────────┐
        │ Loom UI (FastAPI + React)  │   │ MCP server │   │ Recommender    │
        │ subscriber = loom-ui       │   │ 23 tools   │   │ subscriber =   │
        │ scope = read+write *       │   │ subscriber │   │ recommender    │
        │ Sources, Chat, Graph,      │   │ from env   │   │ writes vault   │
        │ Recommender (per cap)      │   │ var        │   │ via materialize│
        └────────────────────────────┘   └────────────┘   └────────────────┘
```

## On-disk layout

```
<storage_root>/
├── .env                       # secrets
├── subscribers.yaml           # permission scope per consumer
├── data/<workspace_id>/
│   ├── workspace.json         # display_name, description, capabilities
│   ├── snapshot.json          # graph snapshot
│   ├── wal.jsonl              # graph write-ahead log
│   ├── paper_registry.json    # ingestion status per paper
│   ├── keyword_index.json     # BM25 corpus
│   ├── semantic_*.{faiss,npy} # FAISS indexes for chunks + propositions
│   ├── chat_history.json      # last N chat messages
│   └── feed.db                # recommender state (only if attached)
└── vault/<workspace_id>/
    ├── *.md                   # ingested papers as markdown
    └── _candidates/           # materialized recommender candidates
```

All durable state is markdown + sqlite + json — readable without loom.

## Selectable per-call paths

| What | Env var | Where | Notes |
|---|---|---|---|
| LLM provider | `LOOM_LLM_PROVIDER` | `loom/llm/__init__.py` | `gemini` / `openrouter` / `mcp_sampling` |
| Embedding provider | `LOOM_LLM_EMBEDDING_PROVIDER` | same | `gemini` (only impl today) |
| Retriever | `LOOM_RETRIEVAL_RETRIEVER` | `loom/retrieval/registry.py` | `adaptive` (default) / `graph_hybrid` / `full_context` |
| Subscriber identity (MCP) | `LOOM_MCP_SUBSCRIBER_ID` | `loom/mcp_server/server.py` | default `claude-code` |
| MCP context window | `LOOM_MCP_CONTEXT_WINDOW` | `loom/llm/mcp_sampling.py` | default 200_000 (Claude host) |

## Request lifecycle

### Chat (UI or MCP `chat_query`)

1. `ChatEngine.chat(message)` appends to history and builds a context query
   from the last 3 user messages.
2. `self.retriever.retrieve(context_query)` returns a `RetrievalResult`
   (chunks, propositions, graph_context, all_results, token estimate,
   retriever_used).
3. Engine assembles a prompt: graph context → propositions → chunk
   previews → conversation tail → user question.
4. `LLMProvider.generate(prompt, model="pro")` returns the answer.
5. Sources are summarized from `all_results[:10]`.
6. Response includes `retriever_used` so the UI / API tells you which
   path ran ("adaptive:full_context", "adaptive:graph_hybrid", etc.).

### Ingestion (MCP `ingest_paper`)

1. `enforce(workspace_id, write=True)` — permission gate.
2. Build an `MCPSamplingLLMProvider` bound to this request's MCP context.
3. `loader.make_pipeline(workspace_id, llm)` — assemble the
   IngestionPipeline.
4. `registry.register_and_queue(identifier)` — paper enters the registry
   as `queued` → `ingesting`.
5. `read_and_ingest_paper(pipeline, identifier)` — chunk → enrich →
   propositions → entity / relationship extraction (all via MCP
   sampling) → graph update.
6. On success: registry → `ingested`; vault markdown written; snapshot
   saved.

### Recommender materialize

1. `materialize_into_workspace(workspace_id, item_ids)` reads the items
   from feed.db.
2. For each: writes `vault/<ws>/_candidates/<item_id>.md` with YAML
   frontmatter (status=candidate, source=recommender, url, kind,
   published_at).
3. Updates `feed_item.status = 'materialized'` so the same candidate
   isn't surfaced again.
4. To bring it into the knowledge graph, run `ingest_paper` on the new
   markdown (or its URL).

## Permissions enforcement

Every workspace-targeting MCP tool calls `enforce(workspace_id, write=)`
at entry. `enforce()` consults the active subscriber's `workspaces` and
`mode`:

```python
if (err := enforce(workspace_id, write=True)) is not None:
    return err
# … real work …
```

A `None` return means access granted. Otherwise the helper returns
`{"ok": False, "error": "permission denied: …"}` which the tool returns
verbatim. The same helper is used across `tools/shared.py`,
`tools/ingestion.py`, `tools/recommender.py`, and `tools/research.py`.

`list_workspaces` is the only tool that *filters* instead of denying: it
returns the subset of workspaces the subscriber can see.

## Adaptive retrieval decision

```python
budget = int(llm.context_window * settings.retrieval.full_context_budget_ratio)
budget = max(0, budget - settings.retrieval.full_context_min_safety_margin_tokens)

if estimated_workspace_tokens <= budget:
    use FullContextRetriever
else:
    use GraphHybridRetriever
```

`estimated_workspace_tokens = Σ tokens(chunk_texts) + Σ tokens(prop_texts)
+ approx_graph_lines`. Recomputed on each call (cheap over in-memory
indexes); add caching later if it ever shows up in profiles.

## Migrations

- `loom/scripts/migrate_drop_kind.py` — strips legacy
  `workspace.json.kind` and (if `feed.db` exists) adds
  `capabilities: ['recommender']`. Idempotent, dry-run by default. Run
  once after upgrading to this branch.

## Adding a new provider / retriever / subscriber

- **LLM provider**: implement the `LLMProvider` Protocol in
  `loom/llm/<name>.py`; register a branch in
  `loom/llm/__init__.py::make_llm_provider`.
- **Retriever**: implement `Retriever` Protocol; add to
  `loom/retrieval/registry.py::RETRIEVERS` and `build_retriever`.
- **Subscriber**: add an entry to `subscribers.yaml`. If invoking via
  MCP, set `LOOM_MCP_SUBSCRIBER_ID=<id>` when launching the server.

## Phase history

Built via the phase plan at
`~/.claude/plans/alright-if-i-were-valiant-cat.md`. Each phase is a
single commit on `refactor/knowledge-substrate`:

- P0 (c0af7bc) Repo merge
- P1 (0b3af85) Provider Protocols
- P2 (e932359) OpenRouter + MCP sampling 1st-class
- P3 (d2d1a37) Retriever Protocol
- P4 (e247dcf) FullContextRetriever + AdaptiveRetriever
- P5 (83d497f) Permissions
- P6 (96ac480) Unified workspace (kind dropped)
- P7 (6d609ca) RecommenderPanel in UI
- P8 retire shim + docs (this commit)
