"""P12 — LLM categorization module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_imports():
    from loom.categorize import (
        DRIFT_THRESHOLD,
        MIN_PAPERS_TO_CATEGORIZE,
        Categorization,
        PaperInput,
        categorization_path,
        generate_categorization,
        is_stale,
        load_categorization,
        save_categorization,
    )
    assert DRIFT_THRESHOLD == 0.20
    assert MIN_PAPERS_TO_CATEGORIZE == 3


def test_is_stale_logic():
    from loom.categorize import Categorization, is_stale

    # No cache + enough papers => stale (regenerate).
    assert is_stale(None, 5) is True
    # No cache but below min => not stale (don't categorize tiny corpora).
    assert is_stale(None, 1) is False
    assert is_stale(None, 2) is False

    cached = Categorization(paper_count=10)
    # 10% drift — under threshold.
    assert is_stale(cached, 11) is False
    assert is_stale(cached, 9) is False
    # 30% drift either direction — over.
    assert is_stale(cached, 13) is True
    assert is_stale(cached, 7) is True
    # Exact 20% drift — under (strict >).
    assert is_stale(cached, 12) is False


def test_save_and_load_roundtrip(tmp_path: Path):
    from loom.categorize import (
        Categorization,
        categorization_path,
        load_categorization,
        save_categorization,
    )

    cat = Categorization(
        version=1,
        generated_at="2026-05-21T10:00:00+00:00",
        model="stub-model",
        paper_count=2,
        summaries={"p1": "s1", "p2": "s2"},
        hierarchy=[{"name": "G", "description": "d", "paper_ids": ["p1", "p2"]}],
    )
    out = save_categorization(tmp_path, cat)
    assert out == categorization_path(tmp_path)
    assert out.exists()

    back = load_categorization(tmp_path)
    assert back is not None
    assert back.model == "stub-model"
    assert back.paper_count == 2
    assert back.summaries == {"p1": "s1", "p2": "s2"}
    assert back.hierarchy[0]["name"] == "G"


def test_load_returns_none_when_missing(tmp_path: Path):
    from loom.categorize import load_categorization
    assert load_categorization(tmp_path) is None


def test_load_returns_none_on_corrupt_file(tmp_path: Path):
    from loom.categorize import categorization_path, load_categorization

    p = categorization_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("not valid json")
    assert load_categorization(tmp_path) is None


def test_extract_json_object_handles_fenced():
    from loom.categorize.categorizer import _extract_json_object

    fenced = '```json\n{"a": 1, "b": [2, 3]}\n```'
    assert _extract_json_object(fenced) == {"a": 1, "b": [2, 3]}

    plain = 'Here is the JSON:\n{"a": 1}'
    assert _extract_json_object(plain) == {"a": 1}


def test_extract_json_object_handles_nested_braces():
    from loom.categorize.categorizer import _extract_json_object

    s = '{"outer": {"inner": {"deep": "ok"}}, "after": 1}'
    parsed = _extract_json_object(s)
    assert parsed["outer"]["inner"]["deep"] == "ok"


def test_extract_json_object_raises_on_garbage():
    from loom.categorize.categorizer import _extract_json_object

    with pytest.raises(ValueError):
        _extract_json_object("no braces at all")


def test_below_min_returns_empty_without_llm():
    """Tiny corpora skip the LLM entirely."""
    from loom.categorize import PaperInput, generate_categorization

    class ExplodingLLM:
        def resolve_model_id(self, role):
            return "stub-pro"

        def generate(self, *a, **kw):
            raise AssertionError("LLM must not be called below MIN_PAPERS_TO_CATEGORIZE")

    result = generate_categorization(ExplodingLLM(), [PaperInput("p1", "Tiny", "")])
    assert result.paper_count == 1
    assert result.hierarchy == []
    assert result.summaries == {}


def test_normal_categorization_via_stub_llm():
    from loom.categorize import PaperInput, generate_categorization
    from loom.llm.provider import LLMResponse

    class StubLLM:
        def resolve_model_id(self, role):
            return "stub-pro"

        def generate(self, prompt, **kw):
            return LLMResponse(
                text=json.dumps({
                    "summaries": {
                        "p1": "Summary 1.",
                        "p2": "Summary 2.",
                        "p3": "Summary 3.",
                    },
                    "hierarchy": [
                        {"name": "Group A", "description": "Da", "paper_ids": ["p1", "p2"]},
                        {"name": "Group B", "description": "Db", "paper_ids": ["p3"]},
                    ],
                }),
                model="stub-pro",
            )

    papers = [PaperInput(f"p{i}", f"Title {i}", f"Abstract {i}") for i in (1, 2, 3)]
    cat = generate_categorization(StubLLM(), papers)
    assert cat.paper_count == 3
    assert cat.model == "stub-pro"
    assert [g["name"] for g in cat.hierarchy] == ["Group A", "Group B"]
    assert sum(len(g["paper_ids"]) for g in cat.hierarchy) == 3
    assert set(cat.summaries.keys()) == {"p1", "p2", "p3"}


def test_orphans_routed_to_uncategorized_group():
    """If the LLM drops papers, they land in a synthetic 'Uncategorized' group."""
    from loom.categorize import PaperInput, generate_categorization
    from loom.llm.provider import LLMResponse

    class ForgetfulLLM:
        def resolve_model_id(self, role):
            return "stub-pro"

        def generate(self, prompt, **kw):
            return LLMResponse(
                text=json.dumps({
                    "summaries": {"p1": "s1"},
                    "hierarchy": [
                        {"name": "G1", "description": "", "paper_ids": ["p1"]},
                    ],
                }),
                model="stub-pro",
            )

    papers = [PaperInput(f"p{i}", f"Title {i}", "") for i in (1, 2, 3)]
    cat = generate_categorization(ForgetfulLLM(), papers)
    names = [g["name"] for g in cat.hierarchy]
    assert "Uncategorized" in names
    uncat = next(g for g in cat.hierarchy if g["name"] == "Uncategorized")
    assert sorted(uncat["paper_ids"]) == ["p2", "p3"]


def test_duplicate_paper_ids_deduplicated():
    """If the LLM lists the same paper twice, it's kept once."""
    from loom.categorize import PaperInput, generate_categorization
    from loom.llm.provider import LLMResponse

    class DupLLM:
        def resolve_model_id(self, role):
            return "stub-pro"

        def generate(self, prompt, **kw):
            return LLMResponse(
                text=json.dumps({
                    "summaries": {},
                    "hierarchy": [
                        {"name": "A", "description": "", "paper_ids": ["p1", "p1", "p2"]},
                        {"name": "B", "description": "", "paper_ids": ["p2", "p3"]},
                    ],
                }),
                model="stub-pro",
            )

    papers = [PaperInput(f"p{i}", f"T{i}", "") for i in (1, 2, 3)]
    cat = generate_categorization(DupLLM(), papers)
    # Each paper appears exactly once across the hierarchy.
    all_ids: list[str] = []
    for g in cat.hierarchy:
        all_ids.extend(g.get("paper_ids", []))
    assert sorted(all_ids) == ["p1", "p2", "p3"]


def test_unknown_paper_ids_dropped():
    """Paper ids the LLM invents (not in input) are dropped."""
    from loom.categorize import PaperInput, generate_categorization
    from loom.llm.provider import LLMResponse

    class HallucinatingLLM:
        def resolve_model_id(self, role):
            return "stub-pro"

        def generate(self, prompt, **kw):
            return LLMResponse(
                text=json.dumps({
                    "summaries": {"p1": "s", "made_up": "should be dropped"},
                    "hierarchy": [
                        {"name": "A", "description": "", "paper_ids": ["p1", "made_up", "p2"]},
                        {"name": "B", "description": "", "paper_ids": ["p3"]},
                    ],
                }),
                model="stub-pro",
            )

    papers = [PaperInput(f"p{i}", f"T{i}", "") for i in (1, 2, 3)]
    cat = generate_categorization(HallucinatingLLM(), papers)
    # Hallucinated ids absent from both summaries and hierarchy.
    assert "made_up" not in cat.summaries
    all_ids: set[str] = set()
    for g in cat.hierarchy:
        all_ids.update(g.get("paper_ids", []))
    assert all_ids == {"p1", "p2", "p3"}


def test_subgroups_supported():
    from loom.categorize import PaperInput, generate_categorization
    from loom.llm.provider import LLMResponse

    class NestedLLM:
        def resolve_model_id(self, role):
            return "stub-pro"

        def generate(self, prompt, **kw):
            return LLMResponse(
                text=json.dumps({
                    "summaries": {},
                    "hierarchy": [
                        {
                            "name": "Top",
                            "description": "",
                            "paper_ids": ["p1"],
                            "subgroups": [
                                {"name": "Sub", "description": "", "paper_ids": ["p2", "p3"]},
                            ],
                        }
                    ],
                }),
                model="stub-pro",
            )

    papers = [PaperInput(f"p{i}", f"T{i}", "") for i in (1, 2, 3)]
    cat = generate_categorization(NestedLLM(), papers)
    top = cat.hierarchy[0]
    assert top["name"] == "Top"
    assert top["paper_ids"] == ["p1"]
    assert "subgroups" in top
    assert top["subgroups"][0]["name"] == "Sub"
    assert sorted(top["subgroups"][0]["paper_ids"]) == ["p2", "p3"]


def test_categorize_module_still_importable():
    """The categorize/ module is still in the tree even though its HTTP
    surface was removed in the unification refactor — the metadata
    worker reuses fit_paper_into_existing for per-document placement.
    """
    from loom.categorize.categorizer import (
        Categorization, PaperInput, generate_categorization, is_stale,
    )
    assert generate_categorization is not None
    assert PaperInput("p1", "T", "abs").paper_id == "p1"
