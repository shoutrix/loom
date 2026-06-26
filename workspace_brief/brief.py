"""Brief data model + LLM-driven generator + persistence."""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loom.llm.base import LLMProvider


# First auto-gen fires once `ingested >= MIN_INGESTED_FOR_FIRST_BRIEF`.
# After that, every AUTO_REGEN_EVERY_N_INGESTED *additional* ingests
# triggers a regen.
MIN_INGESTED_FOR_FIRST_BRIEF = 3
AUTO_REGEN_EVERY_N_INGESTED = 20


@dataclass
class Brief:
    """The structured-fields half of the document. Overwritten on auto-regen."""

    goal: str = ""
    scope: str = ""
    key_questions: list[str] = field(default_factory=list)
    current_focus: str = ""
    exclude: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Brief":
        return cls(
            goal=str(d.get("goal", "") or "").strip(),
            scope=str(d.get("scope", "") or "").strip(),
            key_questions=[
                str(q).strip() for q in (d.get("key_questions") or [])
                if isinstance(q, str) and str(q).strip()
            ],
            current_focus=str(d.get("current_focus", "") or "").strip(),
            exclude=str(d.get("exclude", "") or "").strip(),
        )


@dataclass
class BriefDocument:
    """One workspace's full brief document — auto + user-edited halves."""

    version: int = 1
    generated_at: str = ""
    generated_from_paper_count: int = 0
    model: str = ""
    brief: Brief = field(default_factory=Brief)
    user_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "generated_from_paper_count": self.generated_from_paper_count,
            "model": self.model,
            "brief": self.brief.to_dict(),
            "user_notes": self.user_notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BriefDocument":
        return cls(
            version=int(d.get("version", 1) or 1),
            generated_at=str(d.get("generated_at", "") or ""),
            generated_from_paper_count=int(d.get("generated_from_paper_count", 0) or 0),
            model=str(d.get("model", "") or ""),
            brief=Brief.from_dict(d.get("brief") or {}),
            user_notes=str(d.get("user_notes", "") or ""),
        )


def brief_path(workspace_data_dir: Path) -> Path:
    return Path(workspace_data_dir) / "workspace_brief.json"


def load_brief(workspace_data_dir: Path) -> BriefDocument | None:
    path = brief_path(workspace_data_dir)
    if not path.exists():
        return None
    try:
        return BriefDocument.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def save_brief(workspace_data_dir: Path, doc: BriefDocument) -> Path:
    path = brief_path(workspace_data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def should_regenerate(
    cached: BriefDocument | None, current_ingested: int,
) -> bool:
    """Auto-trigger rule.

    First gen: missing brief AND ingested >= MIN_INGESTED_FOR_FIRST_BRIEF.
    Subsequent: ingested grew by AUTO_REGEN_EVERY_N_INGESTED or more
    since the last generation.
    """
    if current_ingested < MIN_INGESTED_FOR_FIRST_BRIEF:
        return False
    if cached is None:
        return True
    delta = current_ingested - cached.generated_from_paper_count
    return delta >= AUTO_REGEN_EVERY_N_INGESTED


# ----- generator -----------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are an expert technical editor writing a one-page brief for "
    "a researcher's personal knowledge base. Read the papers and their "
    "categorization, then identify what this workspace is for in terms "
    "of goal, scope, key open questions, current focus, and what does "
    "NOT belong. Be concrete and specific; avoid generic phrasing."
)


_USER_PROMPT_TEMPLATE = """Based on the {n_papers} papers below and their categorization, \
write a brief for this workspace.

Output STRICT JSON only — no prose before or after — matching this schema:

{{
  "goal": "<1-2 sentences. What problem or question is this workspace investigating? Concrete.>",
  "scope": "<1-2 sentences. What kinds of papers belong here? Topics, subfields, methods.>",
  "key_questions": [
    "<bullet — an open research question this workspace is tracking>",
    "..."
  ],
  "current_focus": "<1-2 sentences. What's been added recently? What's the active line of inquiry?>",
  "exclude": "<1-2 sentences. What does NOT belong here? Adjacent topics that are deliberately out of scope.>"
}}

Categorization:
{contents_block}

Papers (title + tldr):
{papers_block}
"""


def _build_papers_block(papers: list[dict[str, Any]]) -> str:
    if not papers:
        return "(no papers yet)"
    lines: list[str] = []
    for p in papers:
        title = (p.get("title") or "").strip() or "(untitled)"
        tldr = (p.get("tldr") or "").strip() or "(no tldr)"
        lines.append(f"- {title}\n  {tldr}")
    return "\n".join(lines)


def _build_contents_block(contents_tree: dict[str, Any]) -> str:
    """Compact rendering of the TOC for the prompt."""
    out: list[str] = []

    def _walk(nodes: list[dict[str, Any]], indent: int) -> None:
        for n in nodes:
            prefix = "  " * indent
            out.append(f"{prefix}- {n.get('category', '?')} ({n.get('paper_count', 0)})")
            _walk(n.get("subcategories", []) or [], indent + 1)

    _walk(contents_tree.get("contents", []) or [], 0)
    if contents_tree.get("uncategorized_count"):
        out.append(f"- Uncategorized ({contents_tree['uncategorized_count']})")
    return "\n".join(out) or "(no categories yet)"


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start == -1:
        return {}
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
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def generate_brief(
    llm: "LLMProvider",
    *,
    papers: list[dict[str, Any]],
    contents_tree: dict[str, Any],
    model: str = "pro",
    max_output_tokens: int = 4_096,
) -> Brief:
    """Call the LLM to produce a fresh Brief.

    Caller passes ``papers`` (list of dicts with at least ``title`` and
    ``tldr``) and ``contents_tree`` (the output of
    ``loom.contents.build_contents().to_dict()``). Returns a Brief; on
    any error returns an empty Brief and the caller can decide what to
    do (typically skip persisting until next trigger).
    """
    if not papers:
        return Brief()

    prompt = _USER_PROMPT_TEMPLATE.format(
        n_papers=len(papers),
        papers_block=_build_papers_block(papers),
        contents_block=_build_contents_block(contents_tree),
    )
    try:
        resp = llm.generate(
            prompt,
            model=model,
            system_instruction=_SYSTEM_PROMPT,
            max_output_tokens=max_output_tokens,
            temperature=0.2,
        )
    except Exception:
        return Brief()

    parsed = _extract_json_object(resp.text)
    if not parsed:
        return Brief()
    return Brief.from_dict(parsed)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
