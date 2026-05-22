"""End-to-end smoke: app + MCP server build cleanly."""

from __future__ import annotations

import asyncio


EXPECTED_MCP_TOOLS = {
    # Shared / read
    "list_workspaces", "get_workspace", "list_papers", "get_paper",
    "list_vault_files", "read_vault_file", "health",
    # Vault writes
    "write_vault_note", "write_vault_file",
    # Paper submission (fire-and-forget; loom ingests in the background)
    "submit_paper", "submit_papers",
    "submit_paper_card", "submit_paper_cards",
    "get_paper_card",
    # Bulk-analyze workflow primitives
    "filter_new_papers",
    "build_citation_tree", "get_citation_tree",
}

# Tools that MUST NOT appear on the MCP surface. The underlying code stays
# (HTTP routes and the loom UI use it) but agents don't get it.
DELIBERATELY_EXCLUDED_FROM_MCP = {
    # Search / reasoning — agents do their own search; loom is a
    # deposit channel, not a search proxy.
    "expand_query", "research_search", "chat_query",
    # Recommender — loom-internal, runs on its own schedule.
    "feed_create", "feed_more", "rate_item", "refit_ranker",
    "list_feed_items", "list_feed_runs", "get_feed_profile",
    "get_unrated_items", "propose_description_refresh",
    "update_profile_description", "materialize_into_workspace",
    # Old ingest_paper / ingest_papers (replaced by submit_*).
    "ingest_paper", "ingest_papers",
    # Old list_workspace_papers (replaced by filter_new_papers).
    "list_workspace_papers",
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
        print(f"info: extra tools present beyond expected set: {extra}")
    assert len(names) >= len(EXPECTED_MCP_TOOLS)


def test_excluded_tools_are_not_on_mcp():
    """The MCP surface is narrow by design — search, chat, and
    recommender tools must not be reachable from MCP."""
    from loom.mcp_server.server import build_mcp

    mcp = build_mcp()
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}

    leaked = DELIBERATELY_EXCLUDED_FROM_MCP & names
    assert not leaked, (
        f"these tools should NOT be on the MCP surface: {leaked}"
    )


def test_chat_response_includes_retriever_used():
    """P4 contract: API response model exposes which retriever path ran."""
    from loom.api.routes_chat import ChatResponse

    fields = set(ChatResponse.model_fields.keys())
    assert "retriever_used" in fields
