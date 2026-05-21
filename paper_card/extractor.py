"""
LLM extractor for the 13-field paper card.

One LLM call per paper. Input: the paper's full markdown body (capped
at ~80k characters to fit comfortably even in a 200k-window model
alongside the prompt + output). Output: strict JSON matching the
PaperCard schema. Tolerant parsing handles ```json fences and leading
prose, the same way the categorize/ module does.

Header metadata (title, authors, venue, year, source_url, arxiv_id,
doi) is filled in BY THE CALLER from the paper registry / vault
frontmatter — the LLM only produces the 13 review fields. This keeps
the LLM from hallucinating bibliographic data that's already
authoritative on the source side.

Workspace relevance is the one field that uses workspace context: if
the caller passes a workspace description, the LLM is asked to frame
relevance in those terms. Otherwise it falls back to a generic
"why is this paper interesting" framing.
"""

from __future__ import annotations

import datetime
import json
import re
from typing import TYPE_CHECKING

from loom.paper_card.card import Dataset, PaperCard, RelatedWork

if TYPE_CHECKING:
    from loom.llm.base import LLMProvider


# Hard cap on input markdown to keep prompts bounded.
_MAX_PAPER_CHARS = 80_000

# Hard cap on lists in the LLM output — drop anything beyond this.
_MAX_LIST_ITEMS = 12


_SYSTEM_PROMPT = (
    "You are a research editor producing a structured, dense review of an "
    "academic paper. You read papers like a domain expert: find the actual "
    "contributions, the actual datasets and numbers, the actual limitations. "
    "Avoid generic prose and abstract restatement. Be specific. Output strict "
    "JSON only."
)


def _build_user_prompt(
    paper_text: str,
    *,
    workspace_description: str = "",
) -> str:
    workspace_block = (
        f"\nWorkspace context (use for the workspace_relevance field):\n"
        f"  {workspace_description}\n"
        if workspace_description
        else ""
    )

    return f"""Produce a structured review of the paper below. Output STRICT JSON \
matching exactly this schema (no prose before or after, no fences):

{{
  "tldr": "<1-2 sentences. What this paper is and why it matters.>",
  "problem": "<2-4 sentences. The specific problem being solved.>",
  "approach": "<3-6 sentences. The method, architecture, or algorithm — \
concrete enough that another researcher could sketch it.>",
  "contributions": ["<bullet>", "..."],
  "datasets": [
    {{"name": "<dataset name>", "size": "<e.g. '278 tasks', '50k examples'>", \
"type": "<e.g. 'speech', 'TOD', 'image', 'code'>"}}
  ],
  "setup": "<2-4 sentences. Baselines, metrics, evaluation setup, hardware \
if mentioned.>",
  "results": ["<headline result with the concrete number>", "..."],
  "conclusion": "<2-3 sentences. What did they show.>",
  "strengths": ["<bullet>", "..."],
  "limitations": ["<bullet>", "..."],
  "related_work": [
    {{"title": "<paper title or canonical name>", "why": "<1-line reason it \
matters to this paper>"}}
  ],
  "workspace_relevance": "<2-3 sentences. Why this paper matters for the \
workspace context above. If no workspace context provided, frame it as why \
this paper is interesting in its own right.>",
  "open_questions": ["<bullet>", "..."]
}}

Rules:
- Every list field MUST have at least 1 item if the paper supports it; do \
not invent items.
- Results bullets MUST include concrete numbers where the paper provides \
them (accuracy %, BLEU, MOS, citation count, dataset size, etc.).
- Datasets: list every distinct dataset/benchmark mentioned, with its size \
and type from the paper.
- Related work: 5-10 references that the paper builds on or directly \
contrasts with. Prefer the most influential / most-cited ones.
- Limitations: include both what the authors acknowledge AND what a \
reviewer would flag.
- Be specific. "Uses a transformer" is bad. "12-layer transformer encoder \
with 8 attention heads trained on 1M dialogue turns" is good.
- Output MUST be valid JSON. No trailing commas. Strings must be \
double-quoted. No prose outside the JSON object.
{workspace_block}

Paper:

{paper_text[:_MAX_PAPER_CHARS]}
"""


def _extract_json_object(text: str) -> dict:
    """Tolerant JSON extractor: strip ```json fences, locate balanced { ... }.

    Same approach as categorize/categorizer.py — kept inline so paper_card has
    no dependency on the categorize package.
    """
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
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


def _coerce_str_list(value, *, cap: int = _MAX_LIST_ITEMS) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
        elif isinstance(item, dict):
            for key in ("text", "value", "name"):
                if isinstance(item.get(key), str) and item[key].strip():
                    out.append(item[key].strip())
                    break
    return out[:cap]


def _coerce_datasets(value) -> list[Dataset]:
    if not isinstance(value, list):
        return []
    out: list[Dataset] = []
    for item in value:
        if isinstance(item, dict):
            out.append(Dataset(
                name=str(item.get("name", "") or "").strip(),
                size=str(item.get("size", "") or "").strip(),
                type=str(item.get("type", "") or "").strip(),
            ))
        elif isinstance(item, str) and item.strip():
            out.append(Dataset(name=item.strip()))
    return out[:_MAX_LIST_ITEMS]


def _coerce_related_work(value) -> list[RelatedWork]:
    if not isinstance(value, list):
        return []
    out: list[RelatedWork] = []
    for item in value:
        if isinstance(item, dict):
            out.append(RelatedWork(
                title=str(item.get("title", "") or "").strip(),
                why=str(item.get("why", "") or "").strip(),
            ))
        elif isinstance(item, str) and item.strip():
            out.append(RelatedWork(title=item.strip()))
    return out[:_MAX_LIST_ITEMS]


def generate_paper_card(
    paper_text: str,
    *,
    llm: "LLMProvider",
    paper_id: str,
    header: dict | None = None,
    workspace_description: str = "",
    model: str = "pro",
    max_output_tokens: int = 8192,
) -> PaperCard:
    """Run one LLM call to extract a PaperCard from the paper's markdown.

    Args:
        paper_text: full markdown body of the paper (e.g. from the vault).
        llm: an ``LLMProvider`` (typically Gemini Pro or the active
            provider — picks whichever LLMSettings.provider points to).
        paper_id: registry id of the paper; written into the card.
        header: optional dict of authoritative bibliographic fields:
            ``{title, authors, venue, year, source_url, arxiv_id, doi}``.
            These are NOT LLM-extracted — pass them in from the registry
            so we don't hallucinate them.
        workspace_description: optional description of the workspace
            this paper lives in; if provided, ``workspace_relevance`` is
            framed against it.
        model: 'pro' | 'flash'. Defaults to 'pro' for higher quality
            extraction.
        max_output_tokens: bound on the LLM response.

    Returns:
        A populated ``PaperCard``. If the LLM output fails to parse,
        returns a minimally-filled card with only the header.
    """
    prompt = _build_user_prompt(paper_text, workspace_description=workspace_description)

    resp = llm.generate(
        prompt,
        model=model,
        system_instruction=_SYSTEM_PROMPT,
        max_output_tokens=max_output_tokens,
        temperature=0.2,
    )

    header = header or {}
    base_card = PaperCard(
        version=1,
        generated_at=_now_iso(),
        model=llm.resolve_model_id(model),
        paper_id=paper_id,
        title=str(header.get("title", "")),
        authors=list(header.get("authors", []) or []),
        venue=str(header.get("venue", "")),
        year=header.get("year") if isinstance(header.get("year"), int) else None,
        source_url=str(header.get("source_url", "")),
        arxiv_id=str(header.get("arxiv_id", "")),
        doi=str(header.get("doi", "")),
    )

    try:
        raw = _extract_json_object(resp.text)
    except Exception:
        # Header-only card. Caller can retry via the regenerate path.
        return base_card

    if not isinstance(raw, dict):
        return base_card

    base_card.tldr = str(raw.get("tldr", "")).strip()
    base_card.problem = str(raw.get("problem", "")).strip()
    base_card.approach = str(raw.get("approach", "")).strip()
    base_card.contributions = _coerce_str_list(raw.get("contributions"))
    base_card.datasets = _coerce_datasets(raw.get("datasets"))
    base_card.setup = str(raw.get("setup", "")).strip()
    base_card.results = _coerce_str_list(raw.get("results"))
    base_card.conclusion = str(raw.get("conclusion", "")).strip()
    base_card.strengths = _coerce_str_list(raw.get("strengths"))
    base_card.limitations = _coerce_str_list(raw.get("limitations"))
    base_card.related_work = _coerce_related_work(raw.get("related_work"))
    base_card.workspace_relevance = str(raw.get("workspace_relevance", "")).strip()
    base_card.open_questions = _coerce_str_list(raw.get("open_questions"))
    return base_card


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
