"""
Back-compat shim. The canonical home is loom.llm.mcp_sampling.

This file will be deleted in phase P8 once all import sites have been
updated. For now it re-exports the moved symbols.
"""

from __future__ import annotations

from loom.llm.mcp_sampling import (
    MCPReasoningProvider,
    MCPSamplingLLMProvider,
    _extract_text,
)

__all__ = ["MCPReasoningProvider", "MCPSamplingLLMProvider", "_extract_text"]
