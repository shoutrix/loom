# Loom

A personal **knowledge substrate**: one cloud-portable store, segmented into workspaces, that multiple producers and consumers (your UI, Claude Code, Cursor, the recommender) subscribe to with read/write permissions.

Loom is the union of what used to be two repos — `loom` (FastAPI + React UI + knowledge graph + hybrid retrieval) and `loom-mcp` (MCP server + recommender). They were merged on branch `refactor/knowledge-substrate` (see `~/.claude/plans/alright-if-i-were-valiant-cat.md` for the phase-by-phase history).

## What's inside

| Layer | What it does | Where |
|---|---|---|
| **Storage** | Markdown vault + SQLite + JSON snapshots — boringly portable, fully owned | `data/<ws>/`, `vault/<ws>/` |
| **LLM providers** | Pluggable: Gemini (default), OpenRouter, MCP sampling | `loom/llm/{gemini,openrouter,mcp_sampling}.py` |
| **Embeddings** | Pluggable (Gemini today) | `loom/llm/gemini.py` |
| **Retrieval** | Pluggable + adaptive: full-context when corpus fits, graph-hybrid otherwise | `loom/retrieval/` |
| **Permissions** | YAML-defined per-subscriber scope (workspace ids + read/write) | `loom/permissions/`, `subscribers.yaml` |
| **Ingestion** | Background queue: chunk → enrich → propositions → entity/relationship graph | `loom/ingestion/`, `loom/graph/` |
| **Chat** | Retriever → LLM, with `retriever_used` surfaced for visibility | `loom/chat/` |
| **Recommender** | 3-stage personalized ranker (cold → Bayesian LR + Thompson → LightGBM) | `loom/recommender/` |
| **MCP server** | 23 tools (read, research, ingest, recommender, materialize) | `loom/mcp_server/` |
| **UI** | React + Vite (sources, tabbed editor, graph view, chat, recommender panel) | `loom/frontend/` |

## Requirements

- Python 3.11+
- Node.js 18+ (for the UI)
- A [Google AI (Gemini) API key](https://aistudio.google.com/apikey) (embeddings + default LLM)
- *Optional:* an [OpenRouter](https://openrouter.ai) key if you want a non-Gemini reasoning model

## Quick start

1. Clone the repo. From the **parent** directory:

   ```bash
   cp loom/.env.example loom/.env  # set GEMINI_API_KEY (required)
   ./loom/start.sh
   ```

   Backend on **http://localhost:8788**, UI on **http://localhost:3000**.

2. Optional — register the MCP server with Claude Code in `~/.claude.json`:

   ```json
   {
     "mcpServers": {
       "loom": {
         "command": "/abs/path/to/loom/.venv/bin/python",
         "args": ["-m", "loom.mcp_server.server"],
         "cwd": "/abs/path/to/parent-of-loom",
         "env": {
           "PYTHONPATH": ".",
           "LOOM_MCP_SUBSCRIBER_ID": "claude-code"
         }
       }
     }
   }
   ```

   Then in Claude Code: *"loom: list my workspaces"* → MCP tool call → server returns workspace metadata.

## Provider configuration

The LLM provider is selected by `LOOM_LLM_PROVIDER`:

| Value | Behavior |
|---|---|
| `gemini` (default) | `gemini-2.5-pro` / `gemini-2.0-flash` (1M ctx). Reads `GEMINI_API_KEY`. |
| `openrouter` | Any OpenRouter model. Reads `OPENROUTER_API_KEY`; defaults to `anthropic/claude-sonnet-4.5`. |
| `mcp_sampling` | The MCP client (Claude Code / Desktop) does the reasoning via sampling. No completion key needed on the server. |

Embeddings stay on Gemini regardless (OpenRouter doesn't expose a unified embedding API).

## Retrieval

Three retrievers ship; pick via `LOOM_RETRIEVAL_RETRIEVER`:

- `graph_hybrid` — semantic (FAISS) + keyword (BM25) + graph context, RRF fusion.
- `full_context` — stuff the entire workspace into the model's context window.
- `adaptive` *(default)* — chooses the above per-query based on workspace token count vs the active model's context window (× `LOOM_RETRIEVAL_FULL_CONTEXT_BUDGET_RATIO`, default 0.7).

The active retriever is surfaced as `retriever_used` on every `/chat` response.

## Subscribers

`<storage_root>/subscribers.yaml` declares who can talk to what. Auto-generated on first start with permissive defaults. Edit to lock things down:

```yaml
subscribers:
  - id: loom-ui
    label: Loom Web UI
    workspaces: "*"
    mode: read+write
  - id: claude-code
    label: Claude Code via MCP
    workspaces: ["research-papers", "personal-notes"]
    mode: read+write
  - id: cursor-mcp
    label: Cursor via MCP
    workspaces: ["code-knowledge"]
    mode: read
  - id: recommender
    label: Background recommender
    workspaces: "*"
    mode: read+write
```

The MCP server identifies itself via `LOOM_MCP_SUBSCRIBER_ID`. All workspace-targeting tools call the central `enforce()` helper at entry.

## Recommender

Any workspace can have a recommender attached via the `recommender_create` MCP tool (legacy alias: `feed_create`). It stores state in `<workspace>/feed.db` and surfaces capability `recommender` so the UI shows the "Recs" tab. Use the per-item Materialize button (or the MCP `materialize_into_workspace` tool) to promote ranked candidates into the workspace as documents (vault markdown + paper_registry entry).

## API

FastAPI on `:8788`. Browse `/docs` (Swagger) or `/redoc`. Health: `GET /health`.

## Plan / history

See `~/.claude/plans/alright-if-i-were-valiant-cat.md` for the multi-phase refactor that built this. The `Session log` section records each phase's commit sha and date.
