"""Maximal Marginal Relevance for diversifying a ranked candidate list."""

from __future__ import annotations

import numpy as np

from loom.recommender.features import cosine


def mmr_select(
    candidates: list[tuple[str, float, np.ndarray]],
    *,
    n: int,
    lambda_: float = 0.7,
) -> list[str]:
    """
    Greedy MMR over (id, score, embedding) tuples. Returns selected ids.

    lambda_ closer to 1 favors relevance; closer to 0 favors diversity.
    """
    if not candidates or n <= 0:
        return []
    if len(candidates) <= n:
        return [c[0] for c in candidates]

    # sort by relevance desc
    pool = sorted(candidates, key=lambda c: -c[1])
    selected: list[tuple[str, float, np.ndarray]] = [pool.pop(0)]
    selected_ids = [selected[0][0]]

    while pool and len(selected) < n:
        best_idx = 0
        best_mmr = -1e9
        for i, (cid, rel, emb) in enumerate(pool):
            max_sim = max(cosine(emb, e) for _, _, e in selected)
            mmr = lambda_ * rel - (1.0 - lambda_) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        chosen = pool.pop(best_idx)
        selected.append(chosen)
        selected_ids.append(chosen[0])

    return selected_ids
