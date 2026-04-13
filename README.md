# Loom

Local-first research knowledge system: search and ingest papers, build an LLM-backed knowledge graph, and chat over your library with hybrid retrieval (semantic + BM25 + graph).

## Requirements

- Python 3.11+
- Node.js 18+ (for the UI)
- A [Google AI (Gemini) API key](https://aistudio.google.com/apikey)

## Quick start

1. Clone the repo and enter the `loom` directory.

2. Copy environment config and add your keys:

   ```bash
   cp .env.example .env
   ```

   Edit `.env`: set `GEMINI_API_KEY` (required). Optionally set `LOOM_SEMANTIC_SCHOLAR_API_KEY` for higher Semantic Scholar rate limits.

3. From the **parent** of the `loom` package (the directory that contains the `loom/` folder so `PYTHONPATH=.` resolves), run:

   ```bash
   ./loom/start.sh
   ```

   This creates a Python venv if needed, installs backend + frontend deps, starts the API on **http://localhost:8788**, and the UI on **http://localhost:3000**. Press `Ctrl+C` to stop both.

### Manual run

Backend (run from repo parent so imports work):

```bash
cd /path/to/parent-of-loom
PYTHONPATH=. uvicorn loom.main:app --host 0.0.0.0 --port 8788
```

Frontend:

```bash
cd loom/frontend
npm install
npm run dev
```

## Configuration

Environment variables are read from `loom/.env` (see `.env.example`). Main prefixes:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Required for LLM and embeddings |
| `LOOM_SEMANTIC_SCHOLAR_API_KEY` | Optional; improves S2 rate limits |
| `LOOM_LLM_*` | Model names, temperatures (optional overrides) |
| `LOOM_RATE_*` | LLM rate limits |
| `LOOM_VAULT_DIR` / `LOOM_DATA_DIR` | Root paths for notes and workspace data |
| `LOOM_NEO4J_*` | Optional Neo4j sync |

Workspaces live under `data/<workspace_id>/` (graph snapshot, indexes, registry, chat) and `vault/<workspace_id>/` (markdown notes and ingested documents).

## What’s included

- **Paper search** — Multi-source retrieval with LLM relevance scoring and optional citation-graph expansion.
- **Ingestion** — Background queue for papers, URLs, and uploads; vault-backed markdown; incremental graph updates.
- **Chat** — RAG over chunks, propositions, and graph context (Gemini Pro for answers).
- **UI** — React + Vite: sources, tabbed notes/PDF viewer, resizable panels, 3D knowledge graph, workspace switcher.

## API

FastAPI serves JSON at port **8788** (see `loom/api/`). Health check: `GET /health`.
