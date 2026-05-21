"""
Categorization core.

One LLM call categorizes every non-shortlisted paper in a workspace into a
hierarchy of thematic groups, and produces a 1-line summary per paper.

The LLM is asked for strict JSON. We do a tolerant parse, then validate
that every input paper appears in exactly one (sub)group. Any orphans go
into an auto-generated "Uncategorized" group so the UI never silently
drops papers.

On-disk shape (data/<workspace>/categorization.json):

    {
      "version": 1,
      "generated_at": "2026-05-21T12:34:56+00:00",
      "model": "gemini-2.5-pro",
      "paper_count": 42,
      "summaries": { "<paper_id>": "<one-line summary>" },
      "hierarchy": [
        {
          "name": "Group name",
          "description": "1 sentence describing the group",
          "paper_ids": ["..."],
          "subgroups": [
            { "name": "...", "description": "...", "paper_ids": [...] }
          ]
        }
      ]
    }
"""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loom.llm.base import LLMProvider


DRIFT_THRESHOLD = 0.20  # 20% paper-count change triggers a re-run.

# Minimum papers for a categorization to be worth running. Below this, the UI
# just shows the flat list — no LLM call.
MIN_PAPERS_TO_CATEGORIZE = 3


@dataclass
class PaperInput:
    """One paper handed to the categorizer."""

    paper_id: str
    title: str
    abstract: str = ""


@dataclass
class Categorization:
    """In-memory representation of one workspace's categorization."""

    version: int = 1
    generated_at: str = ""
    model: str = ""
    paper_count: int = 0
    summaries: dict[str, str] = field(default_factory=dict)
    hierarchy: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "model": self.model,
            "paper_count": self.paper_count,
            "summaries": self.summaries,
            "hierarchy": self.hierarchy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Categorization":
        return cls(
            version=int(data.get("version", 1)),
            generated_at=str(data.get("generated_at", "")),
            model=str(data.get("model", "")),
            paper_count=int(data.get("paper_count", 0)),
            summaries=dict(data.get("summaries", {})),
            hierarchy=list(data.get("hierarchy", [])),
        )


def categorization_path(workspace_data_dir: Path) -> Path:
    """Where categorization.json lives for a given workspace data dir."""
    return Path(workspace_data_dir) / "categorization.json"


def load_categorization(workspace_data_dir: Path) -> Categorization | None:
    """Return the cached categorization, or None if missing/corrupt."""
    path = categorization_path(workspace_data_dir)
    if not path.exists():
        return None
    try:
        return Categorization.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def save_categorization(workspace_data_dir: Path, cat: Categorization) -> Path:
    """Write categorization.json (parent dir is created if missing)."""
    path = categorization_path(workspace_data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cat.to_dict(), indent=2), encoding="utf-8")
    return path


def is_stale(
    cached: Categorization | None,
    current_paper_count: int,
    *,
    threshold: float = DRIFT_THRESHOLD,
) -> bool:
    """Return True when the cached categorization should be regenerated.

    Rules:
    - No cache -> stale.
    - cached.paper_count == 0 -> stale if there are any papers now.
    - Else: |current - cached| / cached > threshold.
    """
    if cached is None or cached.paper_count == 0:
        return current_paper_count >= MIN_PAPERS_TO_CATEGORIZE
    drift = abs(current_paper_count - cached.paper_count) / cached.paper_count
    return drift > threshold


# --- prompt + parser ----------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are an expert technical editor organizing a researcher's personal "
    "knowledge base. Given a list of papers (each with a title and abstract), "
    "produce a Wikipedia-style hierarchical categorization. Be concise and "
    "concrete; avoid generic group names."
)


_USER_PROMPT_TEMPLATE = """Organize the following {n_papers} papers into a hierarchical \
categorization.

Rules:
- Produce 3 to 8 top-level groups.
- A group MAY have subgroups if its papers naturally split; do NOT force \
subgroups if a group is small or already coherent.
- Every paper must appear in exactly one (sub)group. No duplicates, no \
omissions.
- Group names: 2-6 words, capitalized like article titles.
- Group descriptions: one sentence each, ≤25 words.
- For each paper, also produce a one-line summary: ≤25 words, factual, \
describing what the paper does or claims (not just the topic).

Output STRICT JSON only — no prose before or after — matching this schema:

{{
  "summaries": {{ "<paper_id>": "<one-line summary>" }},
  "hierarchy": [
    {{
      "name": "<group name>",
      "description": "<one-sentence description>",
      "paper_ids": ["<id>", ...],
      "subgroups": [
        {{
          "name": "...",
          "description": "...",
          "paper_ids": ["<id>", ...]
        }}
      ]
    }}
  ]
}}

`paper_ids` may live at the group level, the subgroup level, or both, but \
each paper id must appear exactly once across the whole hierarchy.

Papers:

{papers_block}
"""


def _build_papers_block(papers: list[PaperInput]) -> str:
    out: list[str] = []
    for p in papers:
        abstract = (p.abstract or "").strip()
        # Cap each abstract so a few long ones don't dominate the prompt.
        if len(abstract) > 1500:
            abstract = abstract[:1500].rstrip() + " …"
        out.append(
            f"paper_id: {p.paper_id}\n"
            f"title: {p.title.strip() or '(untitled)'}\n"
            f"abstract: {abstract or '(no abstract on file)'}\n"
            f"---"
        )
    return "\n".join(out)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM response.

    LLMs occasionally wrap JSON in fences or add a leading sentence even
    when asked not to. We strip ```json fences if present, then locate the
    outermost balanced { ... } block.
    """
    text = text.strip()

    # Strip ``` fences (with or without a language tag).
    fence = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Locate the first { and the matching }.
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")
    depth = 0
    end = -1
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        raise ValueError("Unbalanced braces in LLM JSON response")
    return json.loads(text[start : end + 1])


def _normalize_hierarchy(
    hierarchy: list[dict[str, Any]],
    paper_ids: set[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Walk the hierarchy: collect seen ids, dedupe, drop unknown ids.

    Returns (cleaned_hierarchy, seen_ids).
    """
    seen: set[str] = set()

    def _clean(group: dict[str, Any]) -> dict[str, Any]:
        cleaned_ids: list[str] = []
        for pid in group.get("paper_ids") or []:
            if pid in paper_ids and pid not in seen:
                seen.add(pid)
                cleaned_ids.append(pid)
        sub_in = group.get("subgroups") or []
        sub_cleaned: list[dict[str, Any]] = []
        for sg in sub_in:
            if isinstance(sg, dict):
                sub_cleaned.append(_clean(sg))
        out: dict[str, Any] = {
            "name": str(group.get("name", "Unnamed")).strip() or "Unnamed",
            "description": str(group.get("description", "")).strip(),
            "paper_ids": cleaned_ids,
        }
        if sub_cleaned:
            out["subgroups"] = sub_cleaned
        return out

    cleaned = [_clean(g) for g in hierarchy if isinstance(g, dict)]
    return cleaned, seen


def generate_categorization(
    llm: "LLMProvider",
    papers: list[PaperInput],
    *,
    model: str = "pro",
    max_output_tokens: int = 16_384,
) -> Categorization:
    """Run one LLM call to produce summaries + hierarchy.

    The LLM is asked for strict JSON. We validate that every input paper
    ended up in some (sub)group; missing ones are routed into an
    "Uncategorized" group.
    """
    if len(papers) < MIN_PAPERS_TO_CATEGORIZE:
        # Nothing meaningful to categorize. Caller should fall back to a
        # flat list; we return an empty Categorization tagged at "now".
        return Categorization(
            generated_at=_now_iso(),
            model=llm.resolve_model_id(model),
            paper_count=len(papers),
            summaries={},
            hierarchy=[],
        )

    prompt = _USER_PROMPT_TEMPLATE.format(
        n_papers=len(papers),
        papers_block=_build_papers_block(papers),
    )

    resp = llm.generate(
        prompt,
        model=model,
        system_instruction=_SYSTEM_PROMPT,
        max_output_tokens=max_output_tokens,
        temperature=0.2,
    )
    raw = _extract_json_object(resp.text)

    paper_id_set = {p.paper_id for p in papers}

    # Sanitize summaries: keep only known paper ids; coerce values to short
    # strings.
    summaries: dict[str, str] = {}
    for pid, summary in (raw.get("summaries") or {}).items():
        if pid in paper_id_set and isinstance(summary, str):
            summaries[pid] = summary.strip()

    hierarchy_in = raw.get("hierarchy") or []
    if not isinstance(hierarchy_in, list):
        hierarchy_in = []

    hierarchy, seen = _normalize_hierarchy(hierarchy_in, paper_id_set)

    # Route uncategorized papers into a synthetic group so the UI never
    # silently drops them.
    missing = sorted(paper_id_set - seen)
    if missing:
        hierarchy.append({
            "name": "Uncategorized",
            "description": "Papers the categorizer didn't assign to a group.",
            "paper_ids": missing,
        })

    return Categorization(
        version=1,
        generated_at=_now_iso(),
        model=llm.resolve_model_id(model),
        paper_count=len(papers),
        summaries=summaries,
        hierarchy=hierarchy,
    )


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
