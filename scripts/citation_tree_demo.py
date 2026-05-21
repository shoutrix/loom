"""
Run the citation-tree pipeline against any arXiv paper, end-to-end,
with real Semantic Scholar calls.

Usage:
    cd parent-of-loom && PYTHONPATH=. \
      .venv/bin/python -m loom.scripts.citation_tree_demo \
      [--paper-id ARXIV:2603.13686] [--depth 3] [--no-llm] [--workspace-id demo]

Output:
    - Prints the 5-tier classification (origin / landmark / target /
      convergence / frontier) with paper titles and short stats.
    - Persists the full tree JSON under
      <storage_root>/data/<workspace_id>/citation_trees/<paper_id>.json
      for later re-reading.

The LLM tie-breaker is enabled by default (uses the active provider
per LOOM_LLM_PROVIDER); pass --no-llm to skip it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loom.citation_tree import BuildParams, build_citation_tree
from loom.config import get_settings
from loom.llm import make_llm_provider
from loom.tools.paper_search.sources import SemanticScholarClient


def _short_title(t: str, n: int = 80) -> str:
    t = (t or "(untitled)").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def _print_tier(label: str, ids, tree) -> None:
    print(f"\n=== {label.upper()} ({len(ids)}) ===")
    if not ids:
        print("  (empty)")
        return
    for pid in ids:
        node = tree.subgraph.nodes.get(pid)
        sig = tree.signals.get(pid)
        if node is None:
            print(f"  - {pid}  (no metadata)")
            continue
        year = node.year if node.year is not None else "----"
        cites = node.citation_count
        infl = node.influential_citation_count
        pr = sig.local_pagerank if sig else 0.0
        conv = sig.convergence_count if sig else 0
        score = sig.score_influence if sig else 0.0
        print(
            f"  - [{year}] {cites:>5d} cites ({infl:>3d} infl) "
            f"PR={pr:.4f} conv={conv:>2d}  z={score:+.2f}  {_short_title(node.title)}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", default="ARXIV:2603.13686",
                        help="paper identifier (default: tau-Voice, ARXIV:2603.13686)")
    parser.add_argument("--title", default="",
                        help="title hint for id resolution (optional)")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--per-hop-cap", type=int, default=80)
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--no-llm", action="store_true",
                        help="skip the LLM tie-breaker pass")
    parser.add_argument("--workspace-id", default="demo",
                        help="workspace dir under data/ for persistence")
    args = parser.parse_args(argv)

    settings = get_settings()
    ws_settings = settings.for_workspace(args.workspace_id)
    ws_settings.ensure_dirs()

    print(f"Resolving paper        : {args.paper_id}")
    print(f"Storage root           : {settings.storage_root_dir}")
    print(f"Workspace dir          : {ws_settings.data_dir}")
    print(f"Depth / hop cap / max  : {args.depth} / {args.per_hop_cap} / {args.max_nodes}")
    print(f"LLM tie-breaker        : {'OFF' if args.no_llm else 'ON'}")
    print()

    llm = None
    if not args.no_llm:
        llm = make_llm_provider(settings, workspace_id=args.workspace_id)

    s2 = SemanticScholarClient(api_key=settings.semantic_scholar_api_key or None)

    def progress(step: str, info: dict) -> None:
        info_str = ", ".join(f"{k}={v}" for k, v in info.items())
        print(f"  [{step}] {info_str}", flush=True)

    tree = build_citation_tree(
        args.paper_id,
        ws_settings.data_dir,
        s2_client=s2,
        llm=llm,
        title=args.title,
        params=BuildParams(
            depth=args.depth,
            per_hop_cap=args.per_hop_cap,
            max_nodes=args.max_nodes,
            use_llm_tiebreak=not args.no_llm,
        ),
        progress_cb=progress,
    )

    print("\n" + "=" * 78)
    print(f"Target  : {tree.target.get('title', '')}")
    print(f"          {tree.target.get('paper_id', '')}")
    print(f"Stats   : {tree.stats}")
    print("=" * 78)

    for label in ("origin", "landmark", "target", "convergence", "frontier"):
        _print_tier(label, tree.tiers.get(label, []), tree)

    print("\nSaved to:", ws_settings.data_dir / "citation_trees")
    return 0


if __name__ == "__main__":
    sys.exit(main())
