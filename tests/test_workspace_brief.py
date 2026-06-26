"""P14: tests for the workspace_brief module + auto-regen trigger."""

from __future__ import annotations

import json
from pathlib import Path

from loom.workspace_brief import (
    AUTO_REGEN_EVERY_N_INGESTED,
    MIN_INGESTED_FOR_FIRST_BRIEF,
    Brief,
    BriefDocument,
    brief_path,
    generate_brief,
    load_brief,
    save_brief,
    should_regenerate,
)
from loom.workspace_brief.brief import _extract_json_object


# ----- schema / round-trip ------------------------------------------------


def test_brief_document_roundtrip(tmp_path: Path):
    doc = BriefDocument(
        version=1,
        generated_at="2026-05-23T10:00:00+00:00",
        generated_from_paper_count=8,
        model="gemini-2.5-pro",
        brief=Brief(
            goal="Investigate streaming audio agents.",
            scope="Speech LLMs, voice assistants, dialogue.",
            key_questions=["Q1", "Q2"],
            current_focus="Codec design.",
            exclude="Pure ASR/TTS evaluation.",
        ),
        user_notes="# my notes\n- todo",
    )
    save_brief(tmp_path, doc)
    back = load_brief(tmp_path)
    assert back is not None
    assert back.brief.goal == doc.brief.goal
    assert back.brief.key_questions == ["Q1", "Q2"]
    assert back.user_notes == "# my notes\n- todo"
    assert back.generated_from_paper_count == 8


def test_load_brief_returns_none_when_missing(tmp_path: Path):
    assert load_brief(tmp_path) is None


def test_brief_path_helper(tmp_path: Path):
    assert brief_path(tmp_path) == tmp_path / "workspace_brief.json"


# ----- auto-regen trigger -------------------------------------------------


def test_should_regenerate_below_min_ingested():
    """No brief generated until at least MIN_INGESTED papers exist."""
    for n in range(MIN_INGESTED_FOR_FIRST_BRIEF):
        assert should_regenerate(None, n) is False


def test_should_regenerate_first_gen_at_min_ingested():
    assert should_regenerate(None, MIN_INGESTED_FOR_FIRST_BRIEF) is True
    assert should_regenerate(None, MIN_INGESTED_FOR_FIRST_BRIEF + 10) is True


def test_should_regenerate_after_drift():
    """Regen fires every AUTO_REGEN_EVERY_N_INGESTED additional ingests."""
    cached = BriefDocument(generated_from_paper_count=5)
    # Within the window: no regen.
    for delta in range(AUTO_REGEN_EVERY_N_INGESTED):
        assert should_regenerate(cached, 5 + delta) is False
    # At the window edge: regen.
    assert should_regenerate(cached, 5 + AUTO_REGEN_EVERY_N_INGESTED) is True
    # Far beyond: still regen.
    assert should_regenerate(cached, 5 + AUTO_REGEN_EVERY_N_INGESTED * 3) is True


# ----- generator (LLM stubbed) --------------------------------------------


class _StubLLM:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[str] = []

    def generate(self, prompt: str, **kwargs):
        self.calls.append(prompt)

        class _Resp:
            text = self._response_text
        return _Resp()


def test_generate_brief_parses_structured_response():
    llm = _StubLLM(response_text=json.dumps({
        "goal": "Investigating streaming audio agents.",
        "scope": "Speech LLMs, voice assistants.",
        "key_questions": ["What architectures handle interruption?"],
        "current_focus": "Codec design.",
        "exclude": "Pure ASR/TTS evaluation.",
    }))
    papers = [
        {"title": "P1", "tldr": "tldr 1"},
        {"title": "P2", "tldr": "tldr 2"},
    ]
    contents_tree = {
        "contents": [{"category": "Codec Design", "paper_count": 2}],
        "uncategorized_count": 0,
    }
    brief = generate_brief(llm, papers=papers, contents_tree=contents_tree)
    assert brief.goal.startswith("Investigating")
    assert brief.scope == "Speech LLMs, voice assistants."
    assert brief.key_questions == ["What architectures handle interruption?"]
    assert "Codec Design" in llm.calls[0]  # contents tree was in the prompt
    assert "P1" in llm.calls[0]  # papers were in the prompt


def test_generate_brief_handles_fenced_json():
    llm = _StubLLM(response_text='```json\n{"goal": "g", "scope": "s"}\n```')
    brief = generate_brief(llm, papers=[{"title": "P", "tldr": "t"}], contents_tree={})
    assert brief.goal == "g"
    assert brief.scope == "s"


def test_generate_brief_returns_empty_on_llm_error():
    class _BoomLLM:
        def generate(self, *a, **kw):
            raise RuntimeError("API down")

    brief = generate_brief(_BoomLLM(), papers=[{"title": "P"}], contents_tree={})
    assert brief == Brief()


def test_generate_brief_empty_when_no_papers():
    llm = _StubLLM(response_text="should not be used")
    brief = generate_brief(llm, papers=[], contents_tree={})
    assert brief == Brief()
    assert llm.calls == []  # never called


def test_extract_json_object_tolerates_prose_around():
    text = 'Here you go:\n\n{"goal": "g"} ... and that\'s it.'
    assert _extract_json_object(text) == {"goal": "g"}
