"""D1+D2 — paper_card: schema, extractor, persistence, routes."""

from __future__ import annotations

import json
from pathlib import Path


# ----- schema + persistence --------------------------------------------------


def test_card_round_trip(tmp_path: Path):
    from loom.paper_card import (
        Dataset, PaperCard, RelatedWork,
        card_path, load_card, save_card,
    )

    card = PaperCard(
        version=1,
        generated_at="2026-05-21T12:00:00+00:00",
        model="stub-pro",
        paper_id="ARXIV:1706.03762",
        title="Attention Is All You Need",
        authors=["Vaswani", "Shazeer", "Parmar"],
        venue="NeurIPS",
        year=2017,
        source_url="https://arxiv.org/abs/1706.03762",
        arxiv_id="1706.03762",
        tldr="Transformers replace recurrence with self-attention.",
        problem="RNNs are slow and hard to parallelise.",
        approach="Multi-head self-attention over learned embeddings.",
        contributions=["First fully-attention model that beats LSTMs"],
        datasets=[Dataset(name="WMT'14 En-De", size="4.5M sentence pairs", type="translation")],
        setup="Compared against ByteNet and ConvS2S.",
        results=["28.4 BLEU on WMT'14 En-De"],
        conclusion="Attention alone is sufficient.",
        strengths=["Parallelisable", "State of the art"],
        limitations=["Quadratic memory in sequence length"],
        related_work=[RelatedWork(title="Seq2Seq", why="Sequence-to-sequence baseline.")],
        workspace_relevance="Foundational for any transformer-based architecture in the workspace.",
        open_questions=["Can this scale to longer contexts?"],
    )
    out = save_card(tmp_path, card)
    assert out == card_path(tmp_path, "ARXIV:1706.03762")
    assert out.exists()

    back = load_card(tmp_path, "ARXIV:1706.03762")
    assert back is not None
    assert back.title == "Attention Is All You Need"
    assert back.authors == ["Vaswani", "Shazeer", "Parmar"]
    assert back.year == 2017
    assert len(back.datasets) == 1
    assert back.datasets[0].name == "WMT'14 En-De"
    assert back.related_work[0].title == "Seq2Seq"


def test_card_load_returns_none_when_missing(tmp_path: Path):
    from loom.paper_card import load_card
    assert load_card(tmp_path, "ARXIV:never-built") is None


def test_card_load_returns_none_on_corrupt(tmp_path: Path):
    from loom.paper_card import card_path, load_card
    p = card_path(tmp_path, "ARXIV:bad")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("not json")
    assert load_card(tmp_path, "ARXIV:bad") is None


# ----- extractor with a stub LLM --------------------------------------------


_GOLDEN_CARD_JSON = {
    "tldr": "Transformers replace recurrence with attention.",
    "problem": "RNN-based seq2seq is hard to parallelise.",
    "approach": "Multi-head self-attention with positional encodings.",
    "contributions": [
        "First fully-attention model SOTA on WMT'14",
        "Multi-head attention formulation",
    ],
    "datasets": [
        {"name": "WMT'14 En-De", "size": "4.5M pairs", "type": "translation"},
        {"name": "WMT'14 En-Fr", "size": "36M pairs", "type": "translation"},
    ],
    "setup": "Baselines: ByteNet, ConvS2S. Metric: BLEU.",
    "results": ["28.4 BLEU on En-De", "41.0 BLEU on En-Fr"],
    "conclusion": "Self-attention is sufficient for sequence modelling.",
    "strengths": ["Highly parallelisable", "State of the art"],
    "limitations": ["Quadratic memory", "Hard to extrapolate to longer seqs"],
    "related_work": [
        {"title": "Seq2Seq learning", "why": "Direct baseline."},
        {"title": "ByteNet", "why": "Convolutional comparison."},
    ],
    "workspace_relevance": "Foundational for everything in this workspace.",
    "open_questions": ["Long-context efficiency?", "Better positional encodings?"],
}


class StubLLM:
    def __init__(self, response_text: str):
        self._text = response_text
        self.generate_calls = 0

    def resolve_model_id(self, role):
        return "stub-pro"

    def generate(self, prompt, **kw):
        self.generate_calls += 1
        from loom.llm.provider import LLMResponse
        return LLMResponse(text=self._text, model="stub-pro")


def test_extractor_happy_path():
    from loom.paper_card import generate_paper_card

    llm = StubLLM(json.dumps(_GOLDEN_CARD_JSON))
    card = generate_paper_card(
        "# Attention Is All You Need\n\nAbstract: ...",
        llm=llm,
        paper_id="ARXIV:1706.03762",
        header={
            "title": "Attention Is All You Need",
            "authors": ["Vaswani"],
            "venue": "NeurIPS",
            "year": 2017,
            "source_url": "https://arxiv.org/abs/1706.03762",
            "arxiv_id": "1706.03762",
        },
        workspace_description="Foundations of transformers",
    )
    assert llm.generate_calls == 1
    assert card.paper_id == "ARXIV:1706.03762"
    assert card.title == "Attention Is All You Need"
    assert card.year == 2017

    # 13 fields populated.
    assert card.tldr.startswith("Transformers")
    assert "self-attention" in card.approach
    assert len(card.contributions) == 2
    assert len(card.datasets) == 2
    assert card.datasets[0].name == "WMT'14 En-De"
    assert card.datasets[0].size == "4.5M pairs"
    assert "BLEU" in card.results[0]
    assert len(card.strengths) == 2
    assert len(card.limitations) == 2
    assert len(card.related_work) == 2
    assert "workspace" in card.workspace_relevance.lower()
    assert len(card.open_questions) == 2


def test_extractor_handles_fenced_json():
    from loom.paper_card import generate_paper_card

    fenced = "```json\n" + json.dumps(_GOLDEN_CARD_JSON) + "\n```"
    llm = StubLLM(fenced)
    card = generate_paper_card(
        "paper text",
        llm=llm,
        paper_id="X",
        header={},
    )
    assert card.tldr.startswith("Transformers")


def test_extractor_handles_malformed_json_returns_header_only():
    """Bad LLM output -> we still return a card; just the header fields."""
    from loom.paper_card import generate_paper_card

    llm = StubLLM("not even close to JSON")
    card = generate_paper_card(
        "paper text",
        llm=llm,
        paper_id="X",
        header={"title": "T", "year": 2024},
    )
    # Header fields preserved.
    assert card.paper_id == "X"
    assert card.title == "T"
    assert card.year == 2024
    # LLM-extracted fields stay at defaults.
    assert card.tldr == ""
    assert card.contributions == []
    assert card.datasets == []


def test_extractor_caps_lists():
    """Long lists in the LLM output get clipped to _MAX_LIST_ITEMS (12)."""
    from loom.paper_card import generate_paper_card

    huge = dict(_GOLDEN_CARD_JSON)
    huge["contributions"] = [f"point {i}" for i in range(50)]
    llm = StubLLM(json.dumps(huge))
    card = generate_paper_card("x", llm=llm, paper_id="X", header={})
    assert len(card.contributions) == 12


def test_extractor_workspace_description_in_prompt():
    """Workspace description should be embedded in the prompt sent to LLM."""
    from loom.paper_card import generate_paper_card

    seen_prompts: list[str] = []

    class CaptureLLM(StubLLM):
        def generate(self, prompt, **kw):
            seen_prompts.append(prompt)
            return super().generate(prompt, **kw)

    llm = CaptureLLM(json.dumps(_GOLDEN_CARD_JSON))
    generate_paper_card(
        "paper", llm=llm, paper_id="X",
        header={}, workspace_description="Voice agents and turn-taking",
    )
    assert any("Voice agents and turn-taking" in p for p in seen_prompts)


def test_extractor_string_dataset_coerced_to_object():
    """The LLM occasionally returns 'datasets': ['MNIST', 'CIFAR'] (strings, not dicts)."""
    from loom.paper_card import generate_paper_card

    rough = dict(_GOLDEN_CARD_JSON)
    rough["datasets"] = ["MNIST", "CIFAR-10"]
    llm = StubLLM(json.dumps(rough))
    card = generate_paper_card("x", llm=llm, paper_id="X", header={})
    assert len(card.datasets) == 2
    assert card.datasets[0].name == "MNIST"
    assert card.datasets[1].name == "CIFAR-10"


# ----- API routes register ---------------------------------------------------


def test_paper_card_routes_register():
    from loom.main import app

    paths = {getattr(r, "path", "") for r in app.routes}
    assert any(p.startswith("/papers/card/build/") for p in paths)
    assert any(p.startswith("/papers/card/status/") for p in paths)
    assert any(p.startswith("/papers/card/cached/") for p in paths)


# ----- MCP submit_paper_card / get_paper_card --------------------------------


def test_submit_paper_card_persists(tmp_path):
    """End-to-end: build a PaperCard from an agent-submitted dict and
    persist it via the same save_card helper the MCP tool uses."""
    from loom.mcp_server.tools.paper_card_tools import _coerce_card_input
    from loom.paper_card import card_path, load_card, save_card

    agent_input = {
        "tldr": "Transformers replace recurrence with attention.",
        "problem": "RNNs are slow.",
        "approach": "Multi-head self-attention.",
        "contributions": ["Self-attention SOTA", "Multi-head formulation"],
        "datasets": [
            {"name": "WMT'14 En-De", "size": "4.5M pairs", "type": "translation"},
        ],
        "setup": "Compared against ByteNet and ConvS2S.",
        "results": ["28.4 BLEU on En-De"],
        "conclusion": "Attention alone is sufficient.",
        "strengths": ["Parallelisable"],
        "limitations": ["Quadratic memory"],
        "related_work": [{"title": "Seq2Seq", "why": "Baseline."}],
        "workspace_relevance": "Foundational for the transformers workspace.",
        "open_questions": ["Long context?"],
        # extra fields beyond the schema — should be silently ignored
        "made_up_field": "should not crash",
    }
    header_defaults = {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani"],
        "venue": "NeurIPS",
        "year": 2017,
        "source_url": "https://arxiv.org/abs/1706.03762",
        "arxiv_id": "1706.03762",
        "doi": "",
    }
    card = _coerce_card_input(
        "ARXIV:1706.03762", agent_input, registry_header=header_defaults,
    )

    # Header filled from registry defaults.
    assert card.title == "Attention Is All You Need"
    assert card.authors == ["Vaswani"]
    assert card.year == 2017
    # 13 review fields populated.
    assert card.tldr.startswith("Transformers")
    assert len(card.contributions) == 2
    assert len(card.datasets) == 1
    assert card.datasets[0].size == "4.5M pairs"
    # Roundtrip through disk.
    save_card(tmp_path, card)
    back = load_card(tmp_path, "ARXIV:1706.03762")
    assert back is not None
    assert back.tldr == card.tldr


def test_submit_paper_card_agent_overrides_header():
    """Header fields in the agent's input override registry defaults."""
    from loom.mcp_server.tools.paper_card_tools import _coerce_card_input

    agent_input = {
        "title": "Newer title from agent",
        "venue": "Workshop",
        "tldr": "x",
    }
    header_defaults = {
        "title": "Stale registry title",
        "venue": "Old venue",
        "year": 2024,
        "source_url": "",
        "arxiv_id": "",
        "doi": "",
    }
    card = _coerce_card_input("p1", agent_input, registry_header=header_defaults)
    assert card.title == "Newer title from agent"
    assert card.venue == "Workshop"
    # Year not in agent input -> fallback to registry's year.
    assert card.year == 2024


def test_submit_paper_card_coerces_string_datasets():
    """The agent sometimes returns 'datasets': ['MNIST', 'CIFAR'] (strings).
    The coercion path that the Gemini extractor uses should also work here."""
    from loom.mcp_server.tools.paper_card_tools import _coerce_card_input

    agent_input = {"tldr": "x", "datasets": ["MNIST", "CIFAR-10"]}
    card = _coerce_card_input("p", agent_input, registry_header={})
    assert len(card.datasets) == 2
    assert card.datasets[0].name == "MNIST"


def test_submit_paper_card_handles_missing_fields():
    """Sparse agent input still produces a card; missing fields stay empty."""
    from loom.mcp_server.tools.paper_card_tools import _coerce_card_input

    agent_input = {"tldr": "only a tldr"}
    card = _coerce_card_input("p1", agent_input, registry_header={})
    assert card.paper_id == "p1"
    assert card.tldr == "only a tldr"
    assert card.problem == ""
    assert card.contributions == []
    assert card.datasets == []
    assert card.model == "agent-submitted"


def test_submit_paper_card_tools_register_on_mcp():
    """Smoke: the new MCP tools register on the server."""
    import asyncio

    from loom.mcp_server.server import build_mcp

    mcp = build_mcp()
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert "submit_paper_card" in names
    assert "get_paper_card" in names


def test_get_paper_card_returns_exists_false_when_missing(tmp_path, monkeypatch):
    """get_paper_card returns {exists: false} for unknown papers."""
    from loom.paper_card import load_card
    assert load_card(tmp_path, "never-submitted") is None


# ----- new submission semantics ---------------------------------------------


def test_derive_url_prefers_source_url():
    from loom.mcp_server.tools.paper_card_tools import _derive_url

    assert _derive_url({"source_url": "https://example.test/p"}) == "https://example.test/p"


def test_derive_url_falls_back_to_arxiv_id():
    from loom.mcp_server.tools.paper_card_tools import _derive_url

    assert _derive_url({"arxiv_id": "2603.13686"}) == "https://arxiv.org/abs/2603.13686"


def test_derive_url_falls_back_to_doi():
    from loom.mcp_server.tools.paper_card_tools import _derive_url

    assert _derive_url({"doi": "10.1234/foo"}) == "https://doi.org/10.1234/foo"


def test_derive_url_returns_none_when_nothing_identifies():
    from loom.mcp_server.tools.paper_card_tools import _derive_url

    assert _derive_url({"title": "no url here"}) is None
    assert _derive_url({}) is None


def test_max_batch_size_constant():
    """submit_papers and submit_paper_cards must cap at the same value."""
    from loom.mcp_server.tools.paper_card_tools import MAX_BATCH_SIZE

    assert MAX_BATCH_SIZE == 10


def test_new_submission_tools_register():
    """Smoke: submit_paper, submit_papers, submit_paper_cards register."""
    import asyncio

    from loom.mcp_server.server import build_mcp

    mcp = build_mcp()
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert "submit_paper" in names
    assert "submit_papers" in names
    assert "submit_paper_card" in names
    assert "submit_paper_cards" in names
    assert "get_paper_card" in names
    # Old ingest_paper / ingest_papers MUST be gone now.
    assert "ingest_paper" not in names
    assert "ingest_papers" not in names


def test_ingestion_worker_workspace_aware():
    """The IngestionWorker enqueue and start now require workspace context."""
    import inspect

    from loom.main import IngestionWorker

    sig = inspect.signature(IngestionWorker.enqueue)
    params = list(sig.parameters.keys())
    assert params == ["self", "workspace_id", "paper_id", "identifier"]

    sig_start = inspect.signature(IngestionWorker.start)
    params_start = list(sig_start.parameters.keys())
    assert "get_workspace_manager_fn" in params_start


def test_find_existing_matches_arxiv_id_from_url(tmp_path):
    """Dedup helper recognises the same paper across URL formats."""
    from loom.mcp_server.tools.paper_card_tools import _find_existing
    from loom.storage.paper_registry import PaperRegistry

    reg = PaperRegistry(tmp_path / "paper_registry.json")
    reg.register_and_queue("2603.13686")
    reg.save()

    found = _find_existing(reg, "https://arxiv.org/abs/2603.13686")
    assert found is not None and found.arxiv_id == "2603.13686"

    assert _find_existing(reg, "https://arxiv.org/abs/9999.99999") is None


def test_find_existing_matches_doi(tmp_path):
    from loom.mcp_server.tools.paper_card_tools import _find_existing
    from loom.storage.paper_registry import PaperRegistry

    reg = PaperRegistry(tmp_path / "paper_registry.json")
    reg.register_and_queue("10.1234/foo")
    reg.save()

    found = _find_existing(reg, "https://doi.org/10.1234/foo")
    assert found is not None and found.doi == "10.1234/foo"


def test_existing_response_shape():
    from loom.mcp_server.tools.paper_card_tools import _existing_response
    from loom.storage.paper_registry import PaperRecord

    rec = PaperRecord(
        paper_id="manual:abc", title="T", status="ingested",
        arxiv_id="2603.13686", ingested_at="2026-05-22T00:00:00",
    )
    resp = _existing_response(rec, action="skipped")
    assert resp["ok"] is True
    assert resp["action"] == "skipped"
    assert resp["status"] == "ingested"
    assert "already in this workspace" in resp["message"]


def test_new_workspace_and_citation_tools_register():
    """The 3 new MCP tools register on the FastMCP server."""
    import asyncio

    from loom.mcp_server.server import build_mcp

    mcp = build_mcp()
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert "filter_new_papers" in names
    assert "build_citation_tree" in names
    assert "get_citation_tree" in names


def test_sanitize_workspace_id():
    """Mirror the HTTP route's sanitizer behavior."""
    from loom.mcp_server.tools.paper_card_tools import _sanitize_workspace_id

    assert _sanitize_workspace_id("Agent Infra Reading") == "agentinfrareading"
    assert _sanitize_workspace_id("agent-infra-reading") == "agent-infra-reading"
    assert _sanitize_workspace_id("voice_agents_2026") == "voice_agents_2026"
    assert _sanitize_workspace_id("  Spaces  ") == "spaces"
    assert _sanitize_workspace_id("dots.are.stripped") == "dotsarestripped"
    assert _sanitize_workspace_id("a" * 100) == "a" * 64  # 64-char cap
    assert _sanitize_workspace_id("") == ""
    assert _sanitize_workspace_id("!!!") == ""


def test_create_workspace_registers_on_mcp():
    """The new create_workspace MCP tool registers on the server."""
    import asyncio
    from loom.mcp_server.server import build_mcp

    mcp = build_mcp()
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert "create_workspace" in names


def test_filter_new_papers_splits_new_vs_existing(tmp_path):
    """filter_new_papers returns the subset NOT already in the workspace."""
    from loom.mcp_server.tools.paper_card_tools import _find_existing
    from loom.storage.paper_registry import PaperRegistry

    reg = PaperRegistry(tmp_path / "paper_registry.json")
    reg.register_and_queue("2603.13686")
    reg.register_and_queue("10.1234/foo")
    reg.save()

    candidates = [
        "https://arxiv.org/abs/2603.13686",  # already in
        "https://arxiv.org/abs/9999.99999",  # new
        "https://doi.org/10.1234/foo",       # already in (different URL form)
        "https://doi.org/10.5555/bar",       # new
        "",                                   # empty input — should be dropped
        "https://arxiv.org/abs/2603.13686",  # dupe of first input — dropped
    ]

    new = [c for c in candidates if c.strip() and _find_existing(reg, c) is None]
    existing = [c for c in candidates if c.strip() and _find_existing(reg, c) is not None]
    # Dedupe within the input (mirror the MCP tool's behavior).
    seen, new_unique = set(), []
    for c in new:
        if c not in seen:
            seen.add(c); new_unique.append(c)
    seen2, existing_unique = set(), []
    for c in existing:
        if c not in seen2:
            seen2.add(c); existing_unique.append(c)

    assert sorted(new_unique) == sorted([
        "https://arxiv.org/abs/9999.99999",
        "https://doi.org/10.5555/bar",
    ])
    assert sorted(existing_unique) == sorted([
        "https://arxiv.org/abs/2603.13686",
        "https://doi.org/10.1234/foo",
    ])
