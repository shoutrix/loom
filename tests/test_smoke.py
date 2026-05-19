"""End-to-end smoke: app + MCP server build cleanly."""

from __future__ import annotations

import asyncio


EXPECTED_MCP_TOOLS = {
    # Shared / read
    "list_workspaces", "get_workspace", "list_papers", "get_paper",
    "list_vault_files", "read_vault_file", "health",
    # Vault writes
    "write_vault_note", "write_vault_file",
    # Research
    "expand_query", "research_search",
    # Ingestion
    "ingest_paper", "ingest_papers", "chat_query",
    # Recommender (legacy feed_* names preserved)
    "feed_create", "feed_more", "rate_item", "refit_ranker",
    "list_feed_items", "get_feed_profile",
    "propose_description_refresh", "update_profile_description",
    "get_unrated_items", "list_feed_runs",
    "materialize_into_workspace",
}

# Routes we expect to be present at boot (prefix-level).
EXPECTED_ROUTE_PREFIXES = {
    "/", "/chat", "/docs", "/feed", "/graph", "/health", "/ingest",
    "/openapi.json", "/papers", "/redoc", "/search", "/vault", "/workspaces",
}


def test_fastapi_app_builds():
    from loom.main import app

    prefixes = set()
    for r in app.routes:
        path = getattr(r, "path", "")
        if path.startswith("/"):
            prefix = "/" + path.split("/")[1] if path != "/" else "/"
            prefixes.add(prefix)
    missing = EXPECTED_ROUTE_PREFIXES - prefixes
    assert not missing, f"missing route prefixes: {missing}"


def test_mcp_server_builds_with_all_tools():
    from loom.mcp_server.server import build_mcp

    mcp = build_mcp()
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}

    missing = EXPECTED_MCP_TOOLS - names
    extra = names - EXPECTED_MCP_TOOLS

    assert not missing, f"missing MCP tools: {missing}"
    # If something new shows up, surface it in test output for review (don't fail).
    if extra:
        print(f"info: extra tools present beyond P0-P8 contract: {extra}")
    assert len(names) >= len(EXPECTED_MCP_TOOLS)


def test_chat_response_includes_retriever_used():
    """P4 contract: API response model exposes which retriever path ran."""
    from loom.api.routes_chat import ChatResponse

    fields = set(ChatResponse.model_fields.keys())
    assert "retriever_used" in fields
