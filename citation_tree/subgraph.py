"""
C1: bounded multi-hop subgraph + persistence.

Build a directed subgraph around a target paper P by walking S2's
reference and citation edges. The resulting subgraph is the input to
later phases (signals, centrality, convergence, tier classification).

Direction conventions
---------------------
Edges always go from the *citing* paper to the *cited* paper:

    citing ─→ cited

A "backward" hop from P returns P's references — papers P stands on.
A "forward"  hop from P returns P's citations — papers standing on P.

Each node carries a signed `hop_distance` field:
    -N    = reached only via N backward (reference) hops from the target
     0    = the target itself
    +N    = reached only via N forward (citation) hops from the target
A node reachable in both directions stores the smaller absolute distance.

Caps
----
We hard-cap exploration to keep S2 API usage bounded:
- `depth`: how many hops in EACH direction (default 3).
- `per_hop_cap`: how many papers per direction per hop to consume into
  the next frontier (default 80). Frontier prioritisation is by
  S2's `isInfluential` flag and then by descending edge order.
- `max_nodes`: hard cap on total nodes in the subgraph (default 500).
  Once reached, the BFS stops adding new ids regardless of remaining
  depth.

Persistence
-----------
The subgraph serialises to JSON at
    data/<workspace>/citation_trees/<paper_id>__subgraph.json
matching the design doc's per-workspace storage decision.
"""

from __future__ import annotations

import datetime
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DEPTH = 3
DEFAULT_PER_HOP_CAP = 80
DEFAULT_MAX_NODES = 500

# Per-paper field cap on the abstract — keep node serialisations small
# (a 500-node subgraph at 4 KB each is already 2 MB on disk).
_MAX_ABSTRACT_CHARS = 1200

# Per-direction limit on the references / citations we ask S2 for.
# We further prune to per_hop_cap influential edges after the fetch.
_S2_PAGE_LIMIT = 250


# ----- data model ------------------------------------------------------------


@dataclass
class Edge:
    """A directed citing → cited edge."""

    source: str           # citing paper id
    target: str           # cited paper id
    is_influential: bool = False
    intents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "is_influential": self.is_influential,
            "intents": list(self.intents),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Edge":
        return cls(
            source=str(d["source"]),
            target=str(d["target"]),
            is_influential=bool(d.get("is_influential", False)),
            intents=list(d.get("intents", []) or []),
        )


@dataclass
class Node:
    """A paper in the subgraph."""

    paper_id: str
    title: str = ""
    abstract: str = ""
    year: int | None = None
    venue: str = ""
    citation_count: int = 0
    influential_citation_count: int = 0
    arxiv_id: str = ""
    doi: str = ""
    url: str = ""
    authors: list[dict[str, str]] = field(default_factory=list)
    hop_distance: int = 0          # signed; see module docstring

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "influential_citation_count": self.influential_citation_count,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "url": self.url,
            "authors": list(self.authors),
            "hop_distance": self.hop_distance,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Node":
        return cls(
            paper_id=str(d["paper_id"]),
            title=str(d.get("title", "")),
            abstract=str(d.get("abstract", "")),
            year=d.get("year") if isinstance(d.get("year"), int) else None,
            venue=str(d.get("venue", "")),
            citation_count=int(d.get("citation_count", 0) or 0),
            influential_citation_count=int(d.get("influential_citation_count", 0) or 0),
            arxiv_id=str(d.get("arxiv_id", "")),
            doi=str(d.get("doi", "")),
            url=str(d.get("url", "")),
            authors=list(d.get("authors", []) or []),
            hop_distance=int(d.get("hop_distance", 0) or 0),
        )


@dataclass
class CitationSubgraph:
    """The result of a bounded BFS around a target paper."""

    target_id: str                                 # canonical S2 id of the target
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    version: int = 1

    # ----- queries -----

    def in_degree(self, paper_id: str) -> int:
        """How many edges in the subgraph point TO `paper_id`."""
        return sum(1 for e in self.edges if e.target == paper_id)

    def out_degree(self, paper_id: str) -> int:
        return sum(1 for e in self.edges if e.source == paper_id)

    def backward_nodes(self) -> list[str]:
        """Nodes reached only via reference (backward) hops."""
        return [pid for pid, n in self.nodes.items() if n.hop_distance < 0]

    def forward_nodes(self) -> list[str]:
        return [pid for pid, n in self.nodes.items() if n.hop_distance > 0]

    # ----- serialisation -----

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "target_id": self.target_id,
            "params": self.params,
            "stats": self.stats,
            "nodes": {pid: n.to_dict() for pid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CitationSubgraph":
        return cls(
            version=int(d.get("version", 1)),
            generated_at=str(d.get("generated_at", "")),
            target_id=str(d["target_id"]),
            params=dict(d.get("params", {})),
            stats=dict(d.get("stats", {})),
            nodes={
                str(pid): Node.from_dict(nd)
                for pid, nd in (d.get("nodes") or {}).items()
            },
            edges=[Edge.from_dict(ed) for ed in (d.get("edges") or [])],
        )


# ----- target resolution -----------------------------------------------------


def resolve_target_id(paper_id: str, *, title: str = "", s2_client=None) -> str:
    """Convert any paper identifier into one Semantic Scholar accepts.

    A thin wrapper around the same logic the existing paper_search code
    uses ([tools/paper_search/tool.py::_resolve_s2_id]) — kept independent
    so the citation_tree package doesn't depend on the search pipeline.

    Accepts:
        - raw S2 ids
        - "s2:<id>"
        - "arxiv:<id>" or "ARXIV:<id>"
        - "serper:<arxiv-like-id>"
        - "doi:<doi>"
        - bare URLs (e.g. https://arxiv.org/abs/2603.13686)
        - bare arXiv numbers (e.g. 2603.13686)
    """
    pid = (paper_id or "").strip()
    if not pid:
        return ""

    # URLs: pull arXiv id from the path.
    arxiv_url = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", pid)
    if arxiv_url:
        return f"ARXIV:{arxiv_url.group(1)}"

    if pid.startswith("s2:"):
        return pid[3:]

    if pid.lower().startswith("arxiv:") or pid.lower().startswith("arxiv "):
        rest = pid.split(":", 1)[1] if ":" in pid else pid.split(" ", 1)[1]
        rest = rest.split("v")[0].strip()
        return f"ARXIV:{rest}"

    if pid.startswith("serper:"):
        rest = pid[len("serper:"):]
        if re.match(r"\d{4}\.\d{4,5}", rest):
            return f"ARXIV:{rest.split('v')[0]}"

    if pid.lower().startswith("doi:"):
        return f"DOI:{pid.split(':', 1)[1]}"

    # Bare arXiv-like id.
    m_bare = re.fullmatch(r"\d{4}\.\d{4,5}", pid)
    if m_bare:
        return f"ARXIV:{pid}"

    # Embedded arXiv id (e.g. "manual:2603.13686-vN")
    m_embed = re.search(r"(\d{4}\.\d{4,5})", pid)
    if m_embed:
        return f"ARXIV:{m_embed.group(1)}"

    # Title-search fallback when an S2 client is available.
    if s2_client is not None and title:
        try:
            results = s2_client.search(title, limit=3)
            for r in results:
                if str(r.get("title", "")).lower().strip() == title.lower().strip():
                    s2_pid = str(r.get("id", ""))
                    return s2_pid[3:] if s2_pid.startswith("s2:") else s2_pid
            if results:
                s2_pid = str(results[0].get("id", ""))
                return s2_pid[3:] if s2_pid.startswith("s2:") else s2_pid
        except Exception:
            pass

    return pid


# ----- BFS core --------------------------------------------------------------


def _prioritise(
    raw_edges: list[dict[str, Any]],
    per_hop_cap: int,
) -> list[dict[str, Any]]:
    """Pick which edges to consume into the next frontier.

    Influential edges first, then arbitrary order, capped at per_hop_cap.
    Returns the full list trimmed; the unused edges are still available
    in the subgraph (we don't drop them — they're useful for in-degree
    and centrality later).
    """
    if len(raw_edges) <= per_hop_cap:
        return raw_edges
    influential = [e for e in raw_edges if e.get("isInfluential")]
    non_inf = [e for e in raw_edges if not e.get("isInfluential")]
    return (influential + non_inf)[:per_hop_cap]


def _register_node(
    subgraph: CitationSubgraph,
    paper_id: str,
    *,
    hop_distance: int,
) -> Node:
    """Idempotent node insertion. Keeps the smaller absolute hop_distance."""
    existing = subgraph.nodes.get(paper_id)
    if existing is None:
        node = Node(paper_id=paper_id, hop_distance=hop_distance)
        subgraph.nodes[paper_id] = node
        return node
    if abs(hop_distance) < abs(existing.hop_distance):
        existing.hop_distance = hop_distance
    return existing


def _add_edge(
    subgraph: CitationSubgraph,
    seen_edges: set[tuple[str, str]],
    *,
    source: str,
    target: str,
    is_influential: bool,
    intents: list[str],
) -> None:
    key = (source, target)
    if key in seen_edges:
        return
    seen_edges.add(key)
    subgraph.edges.append(Edge(
        source=source, target=target,
        is_influential=is_influential, intents=list(intents),
    ))


def _populate_metadata(
    subgraph: CitationSubgraph,
    s2_client,
    *,
    batch_size: int = 250,
) -> int:
    """Batch-fetch S2 metadata for every node missing a title.

    Returns the count of nodes successfully enriched.
    """
    needs = [pid for pid, n in subgraph.nodes.items() if not n.title]
    if not needs:
        return 0
    enriched = 0
    for start in range(0, len(needs), batch_size):
        batch = needs[start : start + batch_size]
        try:
            results = s2_client.fetch_paper_batch(batch)
        except Exception:
            results = []
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("paperId", "") or "")
            node = subgraph.nodes.get(pid)
            if node is None:
                continue
            node.title = str(item.get("title", "") or "")
            abstract = str(item.get("abstract", "") or "")
            node.abstract = abstract[:_MAX_ABSTRACT_CHARS]
            node.year = item.get("year") if isinstance(item.get("year"), int) else None
            node.venue = str(item.get("venue", "") or "")
            node.citation_count = int(item.get("citationCount", 0) or 0)
            node.influential_citation_count = int(
                item.get("influentialCitationCount", 0) or 0
            )
            node.url = str(item.get("url", "") or "")
            ext = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
            node.arxiv_id = str(ext.get("ArXiv", "") or "")
            node.doi = str(ext.get("DOI", "") or "")
            raw_authors = item.get("authors") or []
            if isinstance(raw_authors, list):
                node.authors = [
                    {
                        "authorId": str(a.get("authorId", "") or ""),
                        "name": str(a.get("name", "") or ""),
                    }
                    for a in raw_authors
                    if isinstance(a, dict) and a.get("authorId")
                ]
            enriched += 1
    return enriched


def build_subgraph(
    target_paper_id: str,
    s2_client,
    *,
    title: str = "",
    depth: int = DEFAULT_DEPTH,
    per_hop_cap: int = DEFAULT_PER_HOP_CAP,
    max_nodes: int = DEFAULT_MAX_NODES,
    progress_cb=None,
) -> CitationSubgraph:
    """Run bounded BFS in both directions around the target paper.

    Args:
        target_paper_id: any identifier accepted by ``resolve_target_id``.
        s2_client: a ``SemanticScholarClient``-shaped object exposing
            ``fetch_references(pid, rich=True)``,
            ``fetch_citations(pid)``, and ``fetch_paper_batch(ids)``.
        depth: hops in each direction (default 3).
        per_hop_cap: max papers consumed per direction per hop into the
            next frontier (default 80).
        max_nodes: hard ceiling on total nodes (default 500).
        progress_cb: optional ``progress_cb(step_name, info_dict)``.

    Returns:
        ``CitationSubgraph`` with target node + reachable nodes + edges +
        per-node metadata populated. ``params`` and ``stats`` are filled in.
    """
    t0 = time.time()

    resolved = resolve_target_id(target_paper_id, title=title, s2_client=s2_client)
    if not resolved:
        raise ValueError(f"could not resolve paper id: {target_paper_id!r}")

    sg = CitationSubgraph(
        target_id=resolved,
        params={
            "depth": depth,
            "per_hop_cap": per_hop_cap,
            "max_nodes": max_nodes,
            "input_id": target_paper_id,
            "input_title": title,
        },
        generated_at=_now_iso(),
    )
    seen_edges: set[tuple[str, str]] = set()

    # Seed the target at hop 0.
    _register_node(sg, resolved, hop_distance=0)

    api_calls = 0

    # ---- backward BFS (references) ----
    backward_frontier: deque[str] = deque([resolved])
    for hop in range(1, depth + 1):
        if not backward_frontier:
            break
        next_frontier: list[str] = []
        if progress_cb:
            progress_cb("backward_hop_start", {"hop": hop, "frontier_size": len(backward_frontier)})
        while backward_frontier:
            citing = backward_frontier.popleft()
            try:
                refs = s2_client.fetch_references(citing, limit=_S2_PAGE_LIMIT, rich=True)
            except Exception:
                refs = []
            api_calls += 1
            if not isinstance(refs, list):
                continue
            picks = _prioritise(refs, per_hop_cap)
            for r in picks:
                cited_id = str(r.get("paperId", "") or "")
                if not cited_id:
                    continue
                # The full edge list is preserved (not trimmed to picks)
                # because in_degree later wants the full picture.
                pass
            for r in refs:
                cited_id = str(r.get("paperId", "") or "")
                if not cited_id:
                    continue
                _add_edge(
                    sg, seen_edges,
                    source=citing, target=cited_id,
                    is_influential=bool(r.get("isInfluential", False)),
                    intents=r.get("intents", []) or [],
                )
                if cited_id not in sg.nodes and len(sg.nodes) < max_nodes:
                    _register_node(sg, cited_id, hop_distance=-hop)
                    # only the picks advance the frontier
                    if r in picks:
                        next_frontier.append(cited_id)
                elif r in picks and cited_id != resolved:
                    # node already known — re-queue only if it's at this hop
                    next_frontier.append(cited_id)
            if len(sg.nodes) >= max_nodes:
                break
        backward_frontier = deque(next_frontier)
        if len(sg.nodes) >= max_nodes:
            break

    # ---- forward BFS (citations) ----
    forward_frontier: deque[str] = deque([resolved])
    for hop in range(1, depth + 1):
        if not forward_frontier:
            break
        next_frontier = []
        if progress_cb:
            progress_cb("forward_hop_start", {"hop": hop, "frontier_size": len(forward_frontier)})
        while forward_frontier:
            cited = forward_frontier.popleft()
            try:
                cites = s2_client.fetch_citations(cited, limit=_S2_PAGE_LIMIT)
            except Exception:
                cites = []
            api_calls += 1
            if not isinstance(cites, list):
                continue
            picks = _prioritise(cites, per_hop_cap)
            for c in cites:
                citing_id = str(c.get("paperId", "") or "")
                if not citing_id:
                    continue
                _add_edge(
                    sg, seen_edges,
                    source=citing_id, target=cited,
                    is_influential=bool(c.get("isInfluential", False)),
                    intents=c.get("intents", []) or [],
                )
                if citing_id not in sg.nodes and len(sg.nodes) < max_nodes:
                    _register_node(sg, citing_id, hop_distance=hop)
                    if c in picks:
                        next_frontier.append(citing_id)
                elif c in picks and citing_id != resolved:
                    next_frontier.append(citing_id)
            if len(sg.nodes) >= max_nodes:
                break
        forward_frontier = deque(next_frontier)
        if len(sg.nodes) >= max_nodes:
            break

    if progress_cb:
        progress_cb("metadata_start", {"nodes_to_enrich": len(sg.nodes)})
    enriched = _populate_metadata(sg, s2_client)
    api_calls += max(1, (enriched // 250) + 1)  # approximate

    sg.stats = {
        "total_nodes": len(sg.nodes),
        "total_edges": len(sg.edges),
        "backward_nodes": len(sg.backward_nodes()),
        "forward_nodes": len(sg.forward_nodes()),
        "s2_api_calls": api_calls,
        "wall_time_seconds": round(time.time() - t0, 2),
        "metadata_enriched_nodes": enriched,
        "depth": depth,
        "max_nodes_capped": len(sg.nodes) >= max_nodes,
    }
    if progress_cb:
        progress_cb("done", sg.stats)
    return sg


# ----- persistence -----------------------------------------------------------


def _safe_filename(paper_id: str) -> str:
    """Produce a filename-safe slug from an S2 paper id."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)[:120]


def subgraph_path(workspace_data_dir: Path, target_id: str) -> Path:
    return (
        Path(workspace_data_dir)
        / "citation_trees"
        / f"{_safe_filename(target_id)}__subgraph.json"
    )


def save_subgraph(workspace_data_dir: Path, sg: CitationSubgraph) -> Path:
    out = subgraph_path(workspace_data_dir, sg.target_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sg.to_dict(), indent=2), encoding="utf-8")
    return out


def load_subgraph(
    workspace_data_dir: Path,
    target_id: str,
) -> CitationSubgraph | None:
    path = subgraph_path(workspace_data_dir, target_id)
    if not path.exists():
        return None
    try:
        return CitationSubgraph.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# Avoid an "unused import" lint warning on `Iterable` (kept for future use).
_ = Iterable
