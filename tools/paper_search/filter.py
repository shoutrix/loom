from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from loom.tools.paper_search.types import Paper
from loom.tools.paper_search.utils import extract_json_object


def llm_rank_papers(
    llm_provider,
    *,
    query: str,
    papers: list[Paper],
    batch_size: int = 25,
    max_abstract_chars: int = 500,
    max_workers: int = 2,
) -> list[Paper]:
    """LLM-driven relevance judgment.

    Sends batches of (title, abstract) to the LLM and asks it to score each
    paper's relevance to the query on a 0-10 scale.  Papers scoring >= 5 are
    kept; those scoring >= 8 are flagged as "core" (used as seeds for
    multi-hop expansion).

    Returns the papers list filtered and annotated with:
      - llm_relevance  (int 0-10)
      - llm_rationale  (str, one-liner)
    """
    if not papers:
        return []

    batches = [papers[i : i + batch_size] for i in range(0, len(papers), batch_size)]

    all_scores: dict[str, dict[str, Any]] = {}

    def _score_batch(batch: list[Paper]) -> dict[str, dict[str, Any]]:
        return _llm_score_batch(
            llm_provider, query=query, papers=batch,
            max_abstract_chars=max_abstract_chars,
        )

    if len(batches) == 1:
        all_scores = _score_batch(batches[0])
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_score_batch, b): i for i, b in enumerate(batches)}
            for fut in as_completed(futs):
                try:
                    all_scores.update(fut.result())
                except Exception:
                    continue

    kept: list[Paper] = []
    for p in papers:
        pid = str(p.get("id", ""))
        info = all_scores.get(pid)
        if info is None:
            p["llm_relevance"] = 5
            p["llm_rationale"] = "LLM did not return a score; included by default"
            kept.append(p)
            continue
        score = int(info.get("score", 0))
        p["llm_relevance"] = score
        p["llm_rationale"] = str(info.get("rationale", ""))
        if score >= 5:
            kept.append(p)

    kept.sort(key=lambda p: p.get("llm_relevance", 0), reverse=True)
    return kept


def _llm_score_batch(
    llm_provider,
    *,
    query: str,
    papers: list[Paper],
    max_abstract_chars: int,
) -> dict[str, dict[str, Any]]:
    """Ask the LLM to score a batch of papers for relevance."""
    compact = []
    for p in papers:
        abstract = str(p.get("abstract", "") or "")[:max_abstract_chars]
        compact.append({
            "id": p["id"],
            "title": p.get("title", ""),
            "abstract": abstract if abstract else "(no abstract)",
        })

    prompt = (
        "You are a research assistant evaluating papers for relevance to a query.\n\n"
        f"QUERY: {query}\n\n"
        f"PAPERS:\n{json.dumps(compact, indent=1)}\n\n"
        "For EACH paper, assign a relevance score from 0-10:\n"
        "  0-4: Not relevant (different topic, tangentially related at best)\n"
        "  5-6: Somewhat relevant (related background, useful context)\n"
        "  7-8: Relevant (directly addresses the topic or a key sub-problem)\n"
        "  9-10: Highly relevant (core paper for this query)\n\n"
        "When in doubt, lean towards including (score 5+) rather than excluding.\n"
        "Papers that provide important foundational context should score 5-6.\n\n"
        "Return a JSON array:\n"
        '[{"id": "...", "score": 7, "rationale": "one-sentence reason"}]\n'
        "Return ONLY the JSON array, no other text."
    )

    resp = llm_provider.generate(prompt, model="flash", temperature=0.1, max_output_tokens=4096)
    text = resp.text or ""

    parsed = extract_json_object(text)
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict) and "papers" in parsed:
        items = parsed["papers"]
    else:
        items = _try_parse_array(text)

    results: dict[str, dict[str, Any]] = {}
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and "id" in item:
                results[str(item["id"])] = {
                    "score": _clamp_score(item.get("score", 5)),
                    "rationale": str(item.get("rationale", "")),
                }

    return results


def _try_parse_array(text: str) -> list | None:
    """Try to extract a JSON array from text that might have markdown fences."""
    import re
    text = text.strip()
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _clamp_score(val) -> int:
    try:
        s = int(val)
    except (TypeError, ValueError):
        return 5
    return max(0, min(10, s))
