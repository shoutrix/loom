from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from loom.tools.paper_search.types import Paper
from loom.tools.paper_search.utils import extract_json_object

log = logging.getLogger(__name__)


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

    log.info("[LLM Filter] Scoring %d papers in batches of %d (workers=%d)",
             len(papers), batch_size, max_workers)
    t0 = time.time()

    batches = [papers[i : i + batch_size] for i in range(0, len(papers), batch_size)]
    log.info("[LLM Filter] Split into %d batches", len(batches))

    all_scores: dict[str, dict[str, Any]] = {}

    def _score_batch(batch: list[Paper], batch_idx: int) -> dict[str, dict[str, Any]]:
        bt0 = time.time()
        result = _llm_score_batch(
            llm_provider, query=query, papers=batch,
            max_abstract_chars=max_abstract_chars,
        )
        log.info("[LLM Filter] Batch %d: scored %d/%d papers in %.2fs",
                 batch_idx, len(result), len(batch), time.time() - bt0)
        return result

    if len(batches) == 1:
        all_scores = _score_batch(batches[0], 0)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_score_batch, b, i): i for i, b in enumerate(batches)}
            for fut in as_completed(futs):
                batch_idx = futs[fut]
                try:
                    all_scores.update(fut.result())
                except Exception as e:
                    log.warning("[LLM Filter] Batch %d failed: %s", batch_idx, e)

    kept: list[Paper] = []
    dropped = 0
    no_score = 0
    for p in papers:
        pid = str(p.get("id", ""))
        info = all_scores.get(pid)
        if info is None:
            p["llm_relevance"] = 5
            p["llm_rationale"] = "LLM did not return a score; included by default"
            kept.append(p)
            no_score += 1
            continue
        score = int(info.get("score", 0))
        p["llm_relevance"] = score
        p["llm_rationale"] = str(info.get("rationale", ""))
        if score >= 5:
            kept.append(p)
        else:
            dropped += 1

    kept.sort(key=lambda p: p.get("llm_relevance", 0), reverse=True)

    elapsed = round(time.time() - t0, 2)
    score_dist = {}
    for p in kept:
        s = p.get("llm_relevance", 0)
        bucket = f"{s}-{s}"
        score_dist[bucket] = score_dist.get(bucket, 0) + 1

    log.info("[LLM Filter] Done in %.2fs: %d kept, %d dropped (score<5), %d unscored (default 5)",
             elapsed, len(kept), dropped, no_score)
    if kept:
        top3 = kept[:3]
        for p in top3:
            log.info("[LLM Filter]   Top: [%d] %s", p.get("llm_relevance", 0), p.get("title", "")[:80])

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

    from loom.prompts import RELEVANCE_SCORE
    prompt = RELEVANCE_SCORE.format(query=query, papers_json=json.dumps(compact, indent=1))

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
    else:
        log.warning("[LLM Filter] Failed to parse LLM scores from response (%d chars)", len(text))

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
