"""MCP tool registration entrypoint."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from loom.mcp_server.state import MCPState
from loom.mcp_server.workspace import MCPWorkspaceLoader
from loom.mcp_server.tools import documents, shared


def register_all(
    mcp: FastMCP,
    state: MCPState,
    loader: MCPWorkspaceLoader,
) -> None:
    """Register every MCP tool group on the given FastMCP instance.

    Surface after the unification refactor:
      - read-only browsing: list_workspaces, get_workspace, health
      - workspace lifecycle: create_workspace, get_workspace_brief,
        update_workspace_brief, get_workspace_contents
      - document write side (single ingestion door): submit_document,
        submit_documents, get_document, get_document_body,
        list_documents, delete_documents, filter_new_documents

    Deliberately NOT on MCP (the underlying code stays in tree for
    HTTP/UI use):
      - research.expand_query / research_search — agents do their own
        search; loom is a deposit channel, not a search proxy.
      - chat_tools.chat_query — UI uses /chat HTTP route.
      - recommender.* — loom-internal, runs on its own schedule.
    """
    shared.register(mcp, state)
    documents.register(mcp, state, loader)
