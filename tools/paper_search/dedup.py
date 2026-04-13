from __future__ import annotations

import logging
import re

from loom.tools.paper_search.types import Paper

log = logging.getLogger(__name__)


def normalize_title(title: str) -> str:
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _merge_paper(base: Paper, candidate: Paper) -> Paper:
    for fld in ("title", "abstract", "url"):
        if len(str(candidate.get(fld, ""))) > len(str(base.get(fld, ""))):
            base[fld] = candidate.get(fld, "")
    if not base.get("doi") and candidate.get("doi"):
        base["doi"] = candidate["doi"]
    if not base.get("arxiv_id") and candidate.get("arxiv_id"):
        base["arxiv_id"] = candidate["arxiv_id"]
    if not base.get("year") and candidate.get("year"):
        base["year"] = candidate["year"]
    base["citation_count"] = max(
        int(base.get("citation_count", 0) or 0),
        int(candidate.get("citation_count", 0) or 0),
    )
    if not base.get("venue") and candidate.get("venue"):
        base["venue"] = candidate["venue"]
    src = set(base.get("sources", []))
    src.update(candidate.get("sources", []))
    base["sources"] = sorted(src)
    angles = set(base.get("search_angles", []))
    angles.update(candidate.get("search_angles", []))
    base["search_angles"] = sorted(angles)
    if not base.get("authors") and candidate.get("authors"):
        base["authors"] = candidate["authors"]
    return base


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    log.info("[Dedup] Deduplicating %d papers", len(papers))
    by_doi: dict[str, Paper] = {}
    by_arxiv: dict[str, Paper] = {}
    by_title: dict[str, Paper] = {}
    uniques: list[Paper] = []

    doi_merges = 0
    arxiv_merges = 0
    title_merges = 0

    for p in papers:
        doi = str(p.get("doi", "")).strip().lower()
        arxiv_id = str(p.get("arxiv_id", "")).strip().lower()
        ntitle = normalize_title(str(p.get("title", "")))

        target: Paper | None = None
        if doi and doi in by_doi:
            target = by_doi[doi]
            doi_merges += 1
        elif arxiv_id and arxiv_id in by_arxiv:
            target = by_arxiv[arxiv_id]
            arxiv_merges += 1
        elif ntitle and ntitle in by_title:
            target = by_title[ntitle]
            title_merges += 1

        if target is not None:
            _merge_paper(target, p)
            continue

        new_p = dict(p)
        uniques.append(new_p)
        if doi:
            by_doi[doi] = new_p
        if arxiv_id:
            by_arxiv[arxiv_id] = new_p
        if ntitle:
            by_title[ntitle] = new_p

    total_merges = doi_merges + arxiv_merges + title_merges
    log.info("[Dedup] %d → %d unique (merged %d: %d DOI, %d arXiv, %d title)",
             len(papers), len(uniques), total_merges, doi_merges, arxiv_merges, title_merges)
    return uniques
