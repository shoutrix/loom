from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, UTC

from loom.tools.paper_search.types import Paper

CURRENT_YEAR = datetime.now(UTC).year

TIER_1_VENUES = {
    "neurips", "nips", "icml", "iclr", "cvpr", "iccv", "eccv",
    "acl", "emnlp", "naacl", "aaai", "ijcai", "icra", "iros",
    "nature", "science", "cell", "pnas",
    "jmlr", "tmlr", "transactions on pattern analysis",
}

SURVEY_PATTERNS = re.compile(
    r"\b(survey|review|overview|tutorial|systematic review|meta-analysis|"
    r"a comprehensive|state of the art|landscape|taxonomy)\b",
    re.IGNORECASE,
)


def _citation_velocity(paper: Paper) -> float:
    cc = int(paper.get("citation_count", 0) or 0)
    year = paper.get("year")
    if not isinstance(year, int) or year < 1990:
        return float(cc)
    age = max(1, CURRENT_YEAR - year + 1)
    return cc / age


def _venue_score(paper: Paper) -> float:
    venue = str(paper.get("venue", "") or "").lower()
    if not venue:
        return 0.0
    for top in TIER_1_VENUES:
        if top in venue:
            return 1.0
    return 0.3


def _source_diversity_score(paper: Paper) -> float:
    sources = paper.get("sources", [])
    if not isinstance(sources, list):
        return 0.0
    n = len(set(sources))
    if n >= 3:
        return 1.0
    if n == 2:
        return 0.5
    return 0.0


def _is_survey(paper: Paper) -> bool:
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    return bool(SURVEY_PATTERNS.search(text))


def _build_s2_id_map(papers: list[Paper]) -> dict[str, str]:
    """Map S2/arXiv IDs to our paper IDs for cross-referencing."""
    s2_id_to_paper_id: dict[str, str] = {}
    for p in papers:
        pid = str(p.get("id", ""))
        if pid.startswith("s2:"):
            s2_id_to_paper_id[pid[3:]] = pid
        arxiv = str(p.get("arxiv_id", ""))
        if arxiv:
            s2_id_to_paper_id[f"ARXIV:{arxiv}"] = pid
    return s2_id_to_paper_id


def _get_top_s2_candidates(papers: list[Paper], top_k: int) -> list[str]:
    """Get S2 IDs for the top-K papers by citation count."""
    sorted_by_cc = sorted(papers, key=lambda p: int(p.get("citation_count", 0) or 0), reverse=True)
    candidates = []
    for p in sorted_by_cc[:top_k]:
        pid = str(p.get("id", ""))
        if pid.startswith("s2:"):
            candidates.append(pid[3:])
        elif p.get("arxiv_id"):
            candidates.append(f"ARXIV:{p['arxiv_id']}")
    return candidates


def compute_in_set_citation_density(
    papers: list[Paper],
    s2_client,
    top_k: int = 20,
    max_workers: int = 4,
) -> dict[str, int]:
    s2_id_to_paper_id = _build_s2_id_map(papers)
    candidates = _get_top_s2_candidates(papers, top_k)

    if not candidates or s2_client is None:
        return {}

    in_degree: dict[str, int] = {pid: 0 for pid in s2_id_to_paper_id.values()}

    def _fetch_refs(paper_s2_id: str) -> list[str]:
        try:
            return s2_client.fetch_references(paper_s2_id, limit=50)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_fetch_refs, sid): sid for sid in candidates}
        for fut in as_completed(future_map):
            refs = fut.result()
            for ref_id in refs:
                if ref_id in s2_id_to_paper_id:
                    target_pid = s2_id_to_paper_id[ref_id]
                    in_degree[target_pid] = in_degree.get(target_pid, 0) + 1

    return in_degree


def compute_influential_citation_density(
    papers: list[Paper],
    s2_client,
    top_k: int = 20,
    max_workers: int = 4,
) -> dict[str, int]:
    """Like compute_in_set_citation_density but only counts isInfluential=True edges.

    Also uses forward citations (papers citing this one) for richer signal.
    Methodology citations are weighted 2x over background citations.
    """
    s2_id_to_paper_id = _build_s2_id_map(papers)
    candidates = _get_top_s2_candidates(papers, top_k)

    if not candidates or s2_client is None:
        return {}

    influential_degree: dict[str, int] = {pid: 0 for pid in s2_id_to_paper_id.values()}

    def _fetch_influential(paper_s2_id: str) -> list[dict]:
        edges: list[dict] = []
        try:
            refs = s2_client.fetch_references(paper_s2_id, limit=100, rich=True)
            if isinstance(refs, list):
                edges.extend(refs)
        except Exception:
            pass
        try:
            cites = s2_client.fetch_citations(paper_s2_id, limit=100)
            if isinstance(cites, list):
                edges.extend(cites)
        except Exception:
            pass
        return edges

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_fetch_influential, sid): sid for sid in candidates}
        for fut in as_completed(future_map):
            edges = fut.result()
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                if not edge.get("isInfluential", False):
                    continue
                ref_id = edge.get("paperId", "")
                if ref_id in s2_id_to_paper_id:
                    target_pid = s2_id_to_paper_id[ref_id]
                    intents = edge.get("intents", [])
                    weight = 2 if "methodology" in intents else 1
                    influential_degree[target_pid] = influential_degree.get(target_pid, 0) + weight

    return influential_degree


def score_papers(
    papers: list[Paper],
    in_set_citations: dict[str, int] | None = None,
    influential_citations: dict[str, int] | None = None,
    *,
    w_citation_velocity: float = 0.30,
    w_in_set_density: float = 0.20,
    w_influential: float = 0.15,
    w_source_diversity: float = 0.10,
    w_venue: float = 0.15,
    w_survey: float = 0.10,
) -> list[Paper]:
    if not papers:
        return []

    velocities = [_citation_velocity(p) for p in papers]
    max_vel = max(velocities) if velocities else 1.0
    if max_vel == 0:
        max_vel = 1.0

    in_set = in_set_citations or {}
    max_in_set = max(in_set.values()) if in_set else 1
    if max_in_set == 0:
        max_in_set = 1

    influential = influential_citations or {}
    max_influential = max(influential.values()) if influential else 1
    if max_influential == 0:
        max_influential = 1

    for i, paper in enumerate(papers):
        pid = str(paper.get("id", ""))
        vel_norm = velocities[i] / max_vel
        in_set_norm = in_set.get(pid, 0) / max_in_set
        influential_norm = influential.get(pid, 0) / max_influential
        src_score = _source_diversity_score(paper)
        ven_score = _venue_score(paper)
        survey = 1.0 if _is_survey(paper) else 0.0

        score = (
            w_citation_velocity * vel_norm
            + w_in_set_density * in_set_norm
            + w_influential * influential_norm
            + w_source_diversity * src_score
            + w_venue * ven_score
            + w_survey * survey
        )
        paper["importance_score"] = round(score, 4)
        paper["is_survey"] = _is_survey(paper)
        paper["citation_velocity"] = round(velocities[i], 2)
        paper["in_set_citations"] = in_set.get(pid, 0)
        paper["influential_citations"] = influential.get(pid, 0)

    papers.sort(key=lambda p: float(p.get("importance_score", 0.0)), reverse=True)
    return papers
