"""
LLM fitter — drop a single newly-ingested paper into an existing category.

When a user adds a paper via the UI ("+ Add" button), there is no agent
deciding placement. After ingestion, this fitter:
  1. Reads the workspace's existing category paths.
  2. Asks the LLM "does this paper fit any existing path? Return the
     path or null."
  3. Returns the chosen path (or [] if nothing fits).

Critically: the fitter NEVER invents new categories. That's the agent's
job. If no existing path fits, the paper stays Uncategorized — the
user / agent can place it manually later.

Cost: one small LLM call per paper (~500 tokens prompt, JSON response).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loom.llm.base import LLMProvider


_SYSTEM_PROMPT = (
    "You are organizing a researcher's personal knowledge base. Given a "
    "list of EXISTING category paths and one new paper, decide which "
    "existing path the paper fits best. If no existing path is a good "
    "fit, return null. Never invent new categories — that decision "
    "belongs to the human curator."
)


_USER_PROMPT_TEMPLATE = """Existing category paths in this workspace:

{paths_block}

New paper:
  title: {title}
  abstract / tldr: {abstract}

Which existing path best fits this paper?

Output STRICT JSON only — no prose before or after — matching:
{{"path": ["Category", "Subcategory"]}}  // pick one of the listed paths
or
{{"path": null}}  // if no existing path is a good fit

Rules:
- The path MUST be one that appears in the list above, copied exactly.
- Do NOT invent new categories. If nothing fits, return null.
- Prefer the deepest matching path (most specific) when more than one
  matches.
"""


def _build_paths_block(paths: list[list[str]]) -> str:
    if not paths:
        return "(no existing categories — workspace is empty)"
    return "\n".join(
        f"  - {' > '.join(segs)}" for segs in paths
    )


def _extract_json_object(text: str) -> dict:
    """Tolerant JSON extractor: strip ```json fences, take first {…} block."""
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


def fit_paper_into_existing(
    llm: "LLMProvider",
    *,
    title: str,
    abstract: str,
    existing_paths: list[list[str]],
    model: str = "flash",
) -> list[str]:
    """Ask the LLM to place the paper into an existing category path.

    Returns a path (length 1-3) from ``existing_paths``, or [] if nothing
    fits or the workspace has no categories yet. Never invents.

    Errors in the LLM call are non-fatal: returns [] and lets the caller
    log it. The paper stays Uncategorized and can be placed later.
    """
    if not existing_paths:
        return []
    if not title and not abstract:
        return []

    # Build a set of valid paths so we can validate the LLM's choice.
    valid = {tuple(p) for p in existing_paths}

    prompt = _USER_PROMPT_TEMPLATE.format(
        paths_block=_build_paths_block(existing_paths),
        title=title.strip()[:300] or "(no title)",
        abstract=abstract.strip()[:1500] or "(no abstract)",
    )

    try:
        resp = llm.generate(
            prompt,
            model=model,
            system_instruction=_SYSTEM_PROMPT,
            max_output_tokens=256,
            temperature=0.0,
        )
    except Exception:
        return []

    parsed = _extract_json_object(resp.text)
    chosen = parsed.get("path")
    if chosen is None:
        return []
    if not isinstance(chosen, list):
        return []
    cleaned = [str(s).strip() for s in chosen if isinstance(s, str) and str(s).strip()]
    if not cleaned:
        return []
    # Hard validation: only accept a path that's literally in the existing set.
    if tuple(cleaned) not in valid:
        return []
    return cleaned[:3]
