"""End-to-end smoke: FastAPI + MCP server build cleanly with the unified surface."""

from __future__ import annotations

import asyncio


EXPECTED_MCP_TOOLS = {
    # read-only browsing
    "list_workspaces", "get_workspace", "health",
    # workspace lifecycle
    "create_workspace",
    "get_workspace_brief", "update_workspace_brief",
    "get_workspace_contents",
    # unified document write side — ONE ingestion door
    "submit_document", "submit_documents",
    "get_document", "get_document_body",
    "list_documents", "delete_documents",
    "filter_new_documents",
}


# Tools that MUST NOT appear on the MCP surface after the unification refactor.
DELIBERATELY_EXCLUDED_FROM_MCP = {
    # legacy ingestion doors — replaced by submit_document
    "submit_paper_card", "submit_paper_cards", "get_paper_card",
    "submit_document_card", "submit_document_cards", "get_document_card",
    "submit_paper", "submit_papers",
    "ingest_paper", "ingest_papers",
    # legacy vault tools — redundant with get_document / get_document_body;
    # vault is now a pure implementation detail.
    "write_vault_note", "write_vault_file",
    "list_vault_files", "read_vault_file",
    # legacy paper-shape browsing — replaced by list_documents/get_document
    "list_papers", "get_paper", "delete_papers",
    "filter_new_papers", "list_workspace_papers",
    # search / chat / recommender remain off-MCP by design
    "expand_query", "research_search", "chat_query",
    # citation tree — pulled off MCP; the agent doesn't need to know
    # this internal tool exists. Backend code remains for future UI use.
    "build_citation_tree", "get_citation_tree",
    "feed_create", "feed_more", "rate_item", "refit_ranker",
    "list_feed_items", "list_feed_runs", "get_feed_profile",
    "get_unrated_items", "propose_description_refresh",
    "update_profile_description", "materialize_into_workspace",
}


# Route prefixes the FastAPI app must expose.
EXPECTED_ROUTE_PREFIXES = {
    "/", "/chat", "/docs", "/feed", "/graph", "/health",
    "/openapi.json", "/redoc", "/search", "/vault", "/workspaces",
    "/documents",
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

    # Legacy prefixes must NOT come back.
    assert "/papers" not in prefixes, "/papers route surfaced — should be gone"
    assert "/ingest" not in prefixes, "/ingest route surfaced — should be gone"


def test_mcp_server_builds_with_all_tools():
    from loom.mcp_server.server import build_mcp

    mcp = build_mcp()
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}

    missing = EXPECTED_MCP_TOOLS - names
    extra = names - EXPECTED_MCP_TOOLS

    assert not missing, f"missing MCP tools: {missing}"
    if extra:
        print(f"info: extra tools present beyond expected set: {extra}")
    assert len(names) >= len(EXPECTED_MCP_TOOLS)


def test_excluded_tools_are_not_on_mcp():
    """The legacy/parallel ingestion tools must not reappear."""
    from loom.mcp_server.server import build_mcp

    mcp = build_mcp()
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}

    leaked = DELIBERATELY_EXCLUDED_FROM_MCP & names
    assert not leaked, f"these tools should NOT be on the MCP surface: {leaked}"


def test_submit_document_is_only_ingestion_door():
    """Lock the invariant: exactly one document-ingestion tool surface."""
    from loom.mcp_server.server import build_mcp

    mcp = build_mcp()
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    ingestion_tools = {n for n in names if n.startswith("submit_")}
    assert ingestion_tools == {"submit_document", "submit_documents"}, (
        f"expected exactly submit_document(+_documents), got: {ingestion_tools}"
    )


def test_chat_response_includes_retriever_used():
    from loom.api.routes_chat import ChatResponse
    fields = set(ChatResponse.model_fields.keys())
    assert "retriever_used" in fields
