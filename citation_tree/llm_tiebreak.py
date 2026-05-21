"""
C7: LLM tie-breaker pass.

After C2-C5 compute graph signals, there are often a handful of papers
in the middle tiers whose graph signals are nearly indistinguishable.
A single LLM call against the target's title + abstract scores those
ambiguous candidates on topical relevance (0-10). The score is then
fed back into the composite as ``llm_relevance``, with weight 5% in
``score_influence`` (see classifier.py).

Why "tie-breaker" and not "primary scorer"
------------------------------------------
The user's explicit instruction is that a single LLM call is **not
sufficient** for influence judgments — LLMs don't see the citation
graph and tend to bias toward famous papers. So this pass:
- Runs only on AMBIGUOUS papers (those whose influence z-score is
  within a narrow band; default ±0.6 σ).
- Is capped at a small batch size (default 30) to keep cost
  predictable.
- Carries only 5% weight in ``score_influence`` — it can break
  ties between structurally-similar candidates, not override
  structural winners.

Tolerant JSON parsing
---------------------
Same approach as the categorize/ module's ``_extract_json_object`` —
strip ``` fences if present, locate the outermost balanced { … }, and
parse. LLMs occasionally wrap JSON despite instructions.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loom.citation_tree.signals import NodeSignals
    from loom.citation_tree.subgraph import CitationSubgraph
    from loom.llm.base import LLMProvider


# Defaults
DEFAULT_BATCH_SIZE = 30          # max candidates sent to the LLM
DEFAULT_AMBIGUOUS_BAND = 0.6     # ±sigma band on score_influence
DEFAULT_MIN_CANDIDATES = 4       # below this, skip the LLM call entirely


_SYSTEM_PROMPT = (
    "You are an expert technical reviewer. Given a target research paper "
    "(title + abstract) and a list of candidate papers (title + abstract), "
    "score how topically and conceptually relevant each candidate is to "
    "the target's line of research. Output strict JSON only."
)

_USER_PROMPT_TEMPLATE = """Target paper:

  Title: {target_title}
  Abstract: {target_abstract}

Score each of the {n} candidate papers below for relevance to this target's \
line of research. Use a 0-10 scale:

  0  = completely unrelated
  3  = tangentially related (different sub-field but shares a method or \
keyword)
  5  = adjacent (same broad area, different problem)
  7  = closely related (same problem family or same method family)
  10 = directly on the same problem/method as the target

Output STRICT JSON only, matching this schema (no prose, no fences):

{{
  "scores": {{
    "<paper_id>": {{ "score": <float 0-10>, "rationale": "<1 sentence>" }}
  }}
}}

Candidates:

{candidates_block}
"""


def _extract_json_object(text: str) -> dict:
    """Same tolerant parser as categorize/categorizer.py — kept inline so
    citation_tree has no dependency on the categorize package."""
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


def _pick_ambiguous_candidates(
    signals: dict[str, "NodeSignals"],
    *,
    target_id: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    band: float = DEFAULT_AMBIGUOUS_BAND,
) -> list[str]:
    """Return the paper ids in the ±band z-score window for tie-breaking.

    The band is computed on ``score_influence`` since that's where the
    LLM weight lives. Ranks within the band by absolute distance from
    the median (closest-to-the-fence first) so the most ambiguous
    papers get scored first.
    """
    eligible: list[tuple[float, str]] = []
    for pid, sig in signals.items():
        if pid == target_id:
            continue
        if abs(sig.score_influence) <= band:
            eligible.append((abs(sig.score_influence), pid))
    eligible.sort()
    return [pid for _, pid in eligible[:batch_size]]


def _build_candidates_block(
    candidate_ids: list[str],
    subgraph: "CitationSubgraph",
) -> str:
    out: list[str] = []
    for pid in candidate_ids:
        node = subgraph.nodes.get(pid)
        if node is None:
            continue
        abstract = (node.abstract or "").strip()
        if len(abstract) > 800:
            abstract = abstract[:800].rstrip() + " …"
        out.append(
            f"paper_id: {pid}\n"
            f"title: {(node.title or '(untitled)').strip()}\n"
            f"abstract: {abstract or '(no abstract)'}\n"
            f"---"
        )
    return "\n".join(out)


def run_llm_tiebreak(
    subgraph: "CitationSubgraph",
    signals: dict[str, "NodeSignals"],
    llm: "LLMProvider",
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    band: float = DEFAULT_AMBIGUOUS_BAND,
    min_candidates: int = DEFAULT_MIN_CANDIDATES,
) -> dict[str, "NodeSignals"]:
    """Run a single LLM call on the ambiguous middle-tier candidates.

    Mutates ``signals`` in place: each candidate's ``llm_relevance``
    field is set to the LLM's 0-10 score. Non-candidates keep their
    default (0.0). After this returns, the caller should re-run
    ``compute_scores`` so the new llm_relevance values fold into
    ``score_influence``.

    Skips the LLM call entirely if fewer than ``min_candidates``
    papers fall in the ambiguous band.

    Returns the same ``signals`` dict for fluent chaining.
    """
    target = subgraph.nodes.get(subgraph.target_id)
    if target is None:
        return signals

    candidates = _pick_ambiguous_candidates(
        signals, target_id=subgraph.target_id, batch_size=batch_size, band=band,
    )
    if len(candidates) < min_candidates:
        return signals

    candidates_block = _build_candidates_block(candidates, subgraph)
    prompt = _USER_PROMPT_TEMPLATE.format(
        target_title=(target.title or "(untitled target)").strip(),
        target_abstract=(target.abstract or "(no abstract)").strip()[:2000],
        n=len(candidates),
        candidates_block=candidates_block,
    )

    try:
        resp = llm.generate(
            prompt,
            model="pro",
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=8192,
        )
        raw = _extract_json_object(resp.text)
    except Exception:
        # Any failure is non-fatal: the tree still works with graph
        # signals alone. Caller can inspect llm_relevance to see who
        # got scored.
        return signals

    scores_in = raw.get("scores") if isinstance(raw, dict) else None
    if not isinstance(scores_in, dict):
        return signals

    for pid, info in scores_in.items():
        if pid not in signals:
            continue
        if isinstance(info, dict):
            raw_score = info.get("score")
        else:
            raw_score = info
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        signals[pid].llm_relevance = max(0.0, min(10.0, score))
    return signals
