# Loom — Design

> This document explains *why* Loom is shaped the way it is. The
> *what* and *how* live in [`README.md`](README.md) and
> [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 1. What Loom is

Loom is a **personal knowledge substrate**: one cloud-portable store,
segmented into workspaces, that multiple producers and consumers (a web
UI, Claude Code via MCP, Cursor via MCP, a background recommender)
subscribe to with declared read/write scopes.

It is not "another note app." It is not "another RAG." It is the
*storage layer* that those things plug into. The distinction is
load-bearing — most of the design decisions below follow from it.

A three-line elevator pitch:

- **What:** persistent personal knowledge base + retrieval + recommender.
- **How:** plain markdown + SQLite, exposed as MCP tools and an HTTP/UI.
- **Why:** I want the data I read to survive any laptop, any chat tab,
  any vendor — and any agent I use should be able to read and write
  into it.

## 2. Why this project exists

The trigger was prosaic: Loom started as a way to keep a personal
research/notes corpus alive across machines. Company-laptop-issued
tooling (Notion on company SSO, browser-only paid apps, chat-history
running on a hostname I'll lose access to) all evaporate the moment you
return the laptop. A markdown-and-sqlite store inside a cloud-synced
folder doesn't.

That kept the project shape modest until two things shifted:

1. **The substrate vision became practical.** When MCP shipped, there
   was finally a *standard protocol* for arbitrary agents to write into
   a personal store. Before MCP this required bespoke integrations
   (Notion API, Drive API, etc.) that didn't compose. Now a single MCP
   server reaches Claude Code, Claude Desktop, Cursor, and any future
   tool that speaks the protocol.
2. **Long-context models made adaptive retrieval the right default.**
   For modest-sized workspaces — which is most personal corpora — a
   200k–1M context window holds the whole thing. The "must-RAG-because-
   the-corpus-is-big" assumption only kicks in at scale. Loom's
   adaptive retriever encodes that.

Those two shifts re-framed Loom from "a portable notes app" into
"the storage substrate for any agent I trust" — a much bigger swing,
but newly executable.

## 3. Fundamental principles

These are the design commitments. Every architectural choice in §5 is
downstream of one of them.

### 3.1 Substrate-first, not app-first

The knowledge base is the **noun**. The UI, the MCP server, the
recommender, anything Claude or Cursor writes — those are **verbs**
operating on the substrate. This inverts the typical app shape (where
data lives inside the app and gets exported when needed).

Consequences:
- The on-disk format is the public contract, not the API.
- New apps shouldn't require schema migrations to participate.
- A future app I haven't built yet can read the substrate today.

### 3.2 Portable, boring storage

The substrate must be readable without Loom running. Markdown for
documents, SQLite for structured state (ratings, feed history,
graph WAL), JSON for snapshots. No proprietary blobs.

This is the difference between owning your data and renting it. The
choice is deliberately boring — boring formats outlive interesting
tools.

### 3.3 Owner-controlled, cloud-portable

The user picks the cloud. Loom doesn't run a server you sign into; it
expects its `storage_root_dir` to live inside whatever sync mechanism
the user already trusts (Drive desktop, Dropbox, rclone, iCloud Drive).
Sync is *out of scope on purpose* — the harder we'd try to own it, the
worse the lock-in. The job is to keep the format portable so any sync
just works.

### 3.4 Pluggable providers, swappable everywhere

Anything model-vendor-specific must sit behind a Protocol. Today Loom
ships three LLM providers — Gemini, OpenRouter, MCP sampling — and
the chat path doesn't know which one ran. Tomorrow's model swap is one
config line, not a refactor.

This is a bet that the *interface* between code and LLMs (`generate`
with role-based routing) outlives any specific vendor.

### 3.5 Adaptive retrieval over rigid RAG

A 50-paper workspace doesn't need a graph; it fits in Claude
Sonnet's 200k. A 5000-paper workspace doesn't. Hardcoding either
strategy is wrong half the time. The dispatcher picks per query:
`ws_tokens ≤ context_window × 0.7 → full_context, else graph_hybrid`.

This bakes in the observation that long-context models changed the
shape of the trade-off without eliminating RAG entirely.

### 3.6 Producers, consumers, and permissioned subscribers

Every agent that touches the substrate has a declared role:

- **Producers** write documents/items in (ingest_paper,
  recommender's materialize, vault notes from the UI).
- **Consumers** read (chat, list, retrieve).
- **Subscribers** combine the two with a declared scope.

The substrate enforces this at the boundary. An MCP client identified
as `cursor-mcp` with `mode: read` cannot accidentally rewrite the
research workspace even if it's instructed to. The scope is the
contract.

### 3.7 MCP as the universal write channel

Loom doesn't try to own the integrations. It exposes 23 tools over
stdio MCP and trusts any compliant client to drive them. This is the
single most important architectural bet in the codebase:

- **Before MCP**: integrate-per-tool (Notion API, Slack API, Drive
  API). Doesn't compose. Each new agent is a bespoke project.
- **With MCP**: speak one protocol; any client that speaks it can
  read and write.

The substrate becomes valuable in proportion to how many agents speak
into it. MCP makes that scale.

### 3.8 Workspaces as the unit of segmentation

One container shape — a workspace — holds documents, indices, a
graph, optional recommender state. Workspaces are isolated from each
other (separate `data/<ws>/`, separate `vault/<ws>/`), so permissions
and topical separation are the same axis. There are no "research
workspaces" vs "feed workspaces" — that was a P6-era distinction
specifically removed because it conflated *what the data is about*
with *who's writing into it*.

### 3.9 Idempotent, additive migrations

Refactor migrations must be safe to re-run, must back up before
mutating, and must default to dry-run. `migrate_drop_kind.py` is the
template: it backs up `workspace.json`, strips fields additively,
prints a diff in dry-run mode, and exits 0 on a no-op.

The reason: the substrate is the user's owned data. A bad migration
is a worse outcome than a delayed feature.

### 3.10 Tests-as-contract

The 28-test suite added in this refactor isn't there for coverage
metrics. It's there to encode the shape of the interfaces we shipped:
- `LLMProvider` Protocol compliance.
- `Retriever` registry contents.
- `enforce()` semantics across modes/scopes.
- Migration idempotency.

If a refactor breaks any of these, the substrate has broken its
promise to its consumers.

## 4. Influences and prior art

The substrate framing isn't novel. It's a recombination of ideas
that have circulated for decades. The novelty is timing — MCP made
the protocol layer practical, long-context models made the retrieval
trade-off shift, and personal-data-store ergonomics have caught up.

- **[Vannevar Bush — "As We May Think" (1945)](https://web.mit.edu/sts.035/www/PDFs/think.pdf)**.
  The original memex: a personal device that holds your reading,
  forms associative trails, and acts as an external memory. Loom is a
  literal-minded interpretation 80 years later.
- **[Solid Project (Tim Berners-Lee)](https://solidproject.org/)**.
  Pods + Web Access Control. The "your data, your choice of which
  apps read it" framing maps directly to Loom's
  subscriber+permissions model. Loom is what you'd get if you wrote
  a Solid-flavored substrate for a single user, without the RDF.
- **[Local-first software (Ink & Switch, Kleppmann et al.)](https://www.inkandswitch.com/essay/local-first/)**.
  The seven ideals are a checklist Loom is held to: no spinners,
  multi-device, optional network, long-term preservation, security
  by default, user control. Loom's "boring portable format" rule is
  this principle in disguise.
- **[Anytype](https://doc.anytype.io/anytype-docs)**. A working
  end-user product that ships much of the local-first thesis. Useful
  prior art for what users actually want from a substrate; less
  useful as direct inspiration because Anytype owns its data format
  and Loom deliberately doesn't.
- **[Andy Matuschak — evergreen notes](https://notes.andymatuschak.org/Evergreen_notes)**.
  The substrate is only as valuable as what you put into it.
  Matuschak's evergreen note discipline is a reminder that storage is
  necessary but not sufficient — the act of curation is where
  knowledge actually compounds. Loom doesn't enforce this, but the
  recommender's rating loop is a nod in the direction.
- **[Model Context Protocol (Anthropic, 2024)](https://modelcontextprotocol.io/)**.
  The protocol that made the substrate vision newly practical.
  Loom's MCP server is the load-bearing integration surface.

There's a wider conversation about personal data ownership
(NotebookLM, Mem, Reflect, Tana, Heptabase, Logseq) that Loom
intersects with but doesn't try to compete with — they're apps with
a database; Loom is a database with apps.

## 5. Architecture

### 5.1 The substrate

```
<storage_root>/
├── .env                        # secrets
├── subscribers.yaml            # permissions
├── data/<workspace_id>/
│   ├── workspace.json          # metadata + capabilities
│   ├── snapshot.json           # graph snapshot
│   ├── wal.jsonl               # graph write-ahead log
│   ├── paper_registry.json     # ingestion status per paper
│   ├── keyword_index.json      # BM25 corpus
│   ├── semantic_*.{faiss,npy}  # FAISS indices
│   ├── chat_history.json       # recent chat
│   └── feed.db                 # recommender state (if attached)
└── vault/<workspace_id>/
    ├── *.md                    # ingested papers as markdown
    └── _candidates/            # materialized recommender candidates
```

Every artifact is a plain file. Every binary is FAISS or numpy, both
widely readable. The substrate survives `python -m loom` going away.

### 5.2 Pluggable providers

Two Protocols ([`loom/llm/base.py`](loom/llm/base.py)) own the
contract:

```python
class LLMProvider(Protocol):
    usage: UsageTracker
    def set_workspace_context(self, workspace_id: str) -> None: ...
    def generate(self, prompt, *, model="flash", ...) -> LLMResponse: ...
    @property
    def context_window(self) -> int: ...
    def resolve_model_id(self, role: str) -> str: ...

class EmbeddingProvider(Protocol):
    dimensions: int
    model_name: str
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_single(self, text: str) -> np.ndarray: ...
```

Concrete implementations:
- **`GeminiLLMProvider`** — original Loom path, default for compatibility.
- **`OpenRouterLLMProvider`** — OpenAI-compatible HTTP to OpenRouter
  (300+ models behind one key). Static context-window catalogue per
  model id.
- **`MCPSamplingLLMProvider`** — the cleverest piece. When Loom runs
  as an MCP server, the *client's* LLM does the reasoning, forwarded
  back via `ctx.session.create_message`. The server holds no
  completion key — the client's existing Anthropic budget is the
  budget. This is "Claude reasoning over Loom's tools without Loom
  paying for Claude."

Selection is per-config (`LOOM_LLM_PROVIDER`). The chat path doesn't
know which one runs.

### 5.3 Retrieval (adaptive)

Three retrievers, one Protocol
([`loom/retrieval/base.py`](loom/retrieval/base.py)), one registry
([`loom/retrieval/registry.py`](loom/retrieval/registry.py)).

```python
class Retriever(Protocol):
    name: str
    def retrieve(self, query, *, history=None) -> RetrievalResult: ...
```

- **`GraphHybridRetriever`** — semantic (FAISS) + keyword (BM25) +
  graph context, RRF fusion. Loom's original path; kept verbatim.
- **`FullContextRetriever`** — every chunk + every proposition +
  compact graph dump. For tiny corpora.
- **`AdaptiveRetriever`** — wraps the other two, picks per call:

  ```python
  budget = int(llm.context_window * settings.full_context_budget_ratio)
  budget = max(0, budget - settings.full_context_min_safety_margin_tokens)
  if ws_tokens <= budget:
      return full_context.retrieve(query)
  return graph_hybrid.retrieve(query)
  ```

The chat response surfaces `retriever_used = "adaptive:full_context"`
or `"adaptive:graph_hybrid"` so the UI can show which path ran.

This is the principle from §3.5 made concrete. The reason it
defaults to `adaptive` rather than `graph_hybrid` is that most
personal corpora fit; the graph path is the *fallback*, not the
default. Inverting that assumption was a key P4 design move.

### 5.4 Permissions

A YAML file owns the policy
([`subscribers.yaml`](subscribers.yaml), auto-bootstrapped):

```yaml
subscribers:
  - id: loom-ui
    workspaces: "*"
    mode: read+write
  - id: claude-code
    workspaces: ["research", "personal-notes"]
    mode: read+write
  - id: cursor-mcp
    workspaces: ["code-knowledge"]
    mode: read
  - id: recommender
    workspaces: "*"
    mode: read+write
```

Every workspace-targeting MCP tool gates on:

```python
if (err := enforce(workspace_id, write=True)) is not None:
    return err
# … real work …
```

`list_workspaces` *filters* instead of denying — it returns the subset
the subscriber can see.

**Why not OAuth?** OAuth is right for multi-tenant SaaS; it's wrong
for a personal substrate where the user is also the admin. YAML +
env-var identity is faster to reason about, faster to audit, and
doesn't require running an auth server. The trade-off is that
revocation is "edit the file." Acceptable for a personal store.

### 5.5 Producers and consumers

| Subscriber | Role | What it writes / reads |
|---|---|---|
| Loom UI | both | Reads: workspace metadata, chunks, graph, chat. Writes: notes, ingested URLs/files. |
| MCP server (Claude Code) | both | Reads: list/get tools. Writes: ingest_paper, materialize, ratings. |
| MCP server (Cursor, read) | consumer | Reads only. Useful for "find me prior context" without trust to mutate. |
| Recommender | producer | Writes: ranked candidates, ratings → centroids. Reads its own state. |

The **materialize** flow ([`materialize_into_workspace`](loom/mcp_server/tools/recommender.py))
is the bridge between the recommender's transient surface (ranked
items in feed.db) and the workspace's durable surface (markdown +
paper_registry). Writing a recommender candidate into the vault as a
`_candidates/<id>.md` makes it a first-class document — same shape as
anything ingested via ingest_paper.

This is what closes the substrate loop: the recommender isn't a
separate workspace kind anymore (per §3.8). It's a producer that
operates on any workspace and deposits its output into the same
shape every other producer uses.

### 5.6 The knowledge graph (intentionally kept)

Loom builds a graph of entities/relationships/communities during
ingestion ([`loom/graph/`](loom/graph/)). The graph isn't novel — it's
a fairly standard GraphRAG setup — but two design choices are worth
naming:

- **The graph is one retriever among many.** Not the canonical
  retrieval path. The adaptive dispatcher picks the graph when the
  corpus exceeds the context window, not as a religious commitment to
  "graphs are better."
- **The graph is durable** (WAL + snapshot + optional Neo4j sync).
  This means it survives restarts and can be queried independently
  of the LLM. A future "knowledge map" UI or graph-traversal agent
  could be built without any change to ingestion.

The earlier instinct to drop the graph (when it felt unused) was
reversed. The reason: even if humans don't traverse it visually, the
graph context boost in `GraphHybridRetriever` provides measurable
lift for retrieval on large corpora. Keeping the graph durable keeps
that option open.

## 6. Key decisions and trade-offs

These are the choices where the alternative was real, not strawman.

### 6.1 Drop `workspace.kind` vs. keep it

**Chosen:** drop. Every workspace is shape-identical; the recommender
attaches via `capabilities: [recommender]` (derived from the
presence of `feed.db`).

**Why:** the kind flag conflated *what the data is about* with *who
writes into it*. Under the substrate vision, those are orthogonal.
You can have a "research" workspace where the recommender is also
writing candidates, and you can have a "feed" workspace that someone
also chats over. The two flows were already overlapping; the flag
just blocked them from composing.

**Cost:** a one-time migration (`migrate_drop_kind.py`). Idempotent,
dry-run by default.

### 6.2 Merge `loom-mcp` into `loom` vs. shared library

**Chosen:** merge into one repo.

**Why:** the fork existed because of how MCP support was iterated on,
not because there was a clean separation. Most of the code was
shared. A "shared kbase library + two thin shells" would have been
cleaner *in theory* — and one whole new package to maintain *in
practice*. For a single-user project, one repo wins.

**Cost:** the `loom-mcp` directory is gone (deliberately). Anyone
who had a checkout of it needs to migrate to `refactor/knowledge-substrate`.

### 6.3 Adaptive retrieval as the default

**Chosen:** `retriever = adaptive` (not `graph_hybrid`).

**Why:** corpora are smaller than people assume, and 200k–1M context
windows are larger than retrieval defaults assume. The graph path is
the fallback because the *common* case (personal-scale workspaces)
fits in context.

**Cost:** full-context retrieval consumes more tokens per query.
That's mitigated by `full_context_budget_ratio = 0.7` (don't pack to
the brim) and the safety margin (8k tokens reserved for the prompt
itself + answer). The cost is measured in tokens, not in quality.

### 6.4 YAML subscribers vs. OAuth

**Chosen:** YAML file, env-var identity.

**Why:** the user is the admin. OAuth's complexity earns its keep
only when you have actual third-party clients you don't control.
For a personal store driven by clients you launched yourself, a
file you can `cat` and `vim` is strictly better.

**Cost:** the scope file is not cryptographically tied to the
running process. A misbehaving MCP client *could* set
`LOOM_MCP_SUBSCRIBER_ID=loom-ui` and gain full access. Acceptable
threat model: clients launched on your machine, by you, are
trusted-by-default.

### 6.5 Materialize as an explicit step

**Chosen:** the recommender produces ranked candidates; the user
(or an agent) explicitly *materializes* them into the workspace.

**Why:** auto-materializing every recommended item would conflate
"the recommender thinks you might like this" with "this is in your
knowledge base." Curation is a human-meaningful boundary.

**Cost:** an extra step. Worth it.

## 7. What's deliberately out of scope

- **Cloud sync.** Drive desktop / rclone / iCloud handle it.
  Owning sync would mean owning conflict resolution would mean owning
  a server. No.
- **Multi-user collaboration.** Loom is single-user. Multi-user
  would change every assumption — auth, conflict resolution, scope
  inheritance, audit logging. A different project.
- **Mobile native.** The web UI renders fine on mobile browsers.
  A native app is a different codebase for marginal benefit.
- **Built-in evaluation framework.** Eval rigor for the chat/retrieval
  path will become important; for now, manual A/B between providers
  and retriever-used tags in the response are the lightweight
  substitute.
- **Production multi-tenant deployment.** Loom is not a SaaS product
  shape. If you want to host it, you host one instance per user with
  one storage root per user.

## 8. Operating model

The plan file
[`~/.claude/plans/alright-if-i-were-valiant-cat.md`](~/.claude/plans/alright-if-i-were-valiant-cat.md)
is the state machine for the multi-phase refactor that built this.
Each phase is one commit; the session log tracks what shipped when.

Going forward, the same discipline applies:

- **One responsibility per commit.** P0–P8 each landed independently
  green.
- **Tests as the contract.** The 28-test suite ([`tests/`](tests/))
  is the floor; extend it as you extend the surface.
- **Idempotent migrations.** Every schema change ships with a script
  that's safe to re-run, defaults to dry-run, backs up before
  mutating.
- **Subscriber-aware refactors.** Don't break a subscriber's scope
  contract without a deprecation cycle (the kept `feed_*` MCP names
  alongside `recommender_*` is the template).

## 8.5 Citation tree (built 2026-05-21)

Multi-hop citation analysis for any target paper. Bounded BFS in
both directions (default depth 3) builds a 200–500 node subgraph;
per-node signals (PageRank, time-balanced PageRank, convergence
counts, S2's `isInfluential`, methodology-citation ratio, citation
velocity) feed a composite score; the LLM is a 5%-weight
tie-breaker on ambiguous middle-tier candidates. Output is a
5-tier tree: origin / landmark / target / convergence / frontier.

Lives in [`loom/citation_tree/`](loom/citation_tree/). Design + phase
history at `~/.claude/plans/citation-tree-design.md`. The user-facing
surface is the "Citation tree" button in PaperViewer; the API is
`POST /papers/citation-tree/start` + `GET /papers/citation-tree/{...}`.

The key principle: a single LLM call is insufficient for influence
judgment because LLMs don't see the citation graph. Aggregating
multiple structural signals (PageRank, convergence, methodology
ratio) and using the LLM only to break ties matches the literature
on milestone-paper identification (Mariani et al. 2016; the
"Promise and Pitfalls" paper on PageRank for citations).

## 9. Future work

Listed in rough priority order.

1. **HTTP route for materialize.** The UI button currently copies an
   MCP command to the clipboard; a direct `/feed/materialize` route
   would close the loop without a clipboard handoff. (Note: vault-
   write MCP tools — `write_vault_note`, `write_vault_file` — shipped
   separately, closing the "agent composes synthesis and deposits it"
   gap that was previously listed here.)
2. **Embedding provider abstraction completion.** Gemini is the only
   embedder today. Adding Voyage and Cohere with a `(model, dim)`
   composite cache key (rather than just `model_name`) defends
   against silent shape mismatches on switchover.
3. **Re-embedding script.** When you swap embedders, you need to
   rebuild the FAISS index from cached chunk texts. A
   `loom.scripts.reembed_workspace` script is a half-day's work.
4. **Recommender as a standalone daemon.** Today the recommender
   runs inside the MCP request. A separate cron-style process that
   wakes up daily, runs `feed_more` for every recommender-enabled
   workspace, and (optionally) pings Slack/email with a digest is the
   natural next step. Mentioned as P9 in the plan, deliberately
   deferred.
5. **Graph improvements.** The current graph is fine; a better
   community summarization pass and a "bridge entity" UI affordance
   would make the graph visually useful in addition to behind-the-
   scenes useful.
6. **Per-tool deny-lists.** The permission registry supports
   `tools_allow` / `tools_deny` per subscriber but the enforce path
   doesn't consult them yet. Wiring them in is a one-screen
   refactor.
7. **Audit log.** Append-only log of every MCP tool call (subscriber,
   tool, workspace, args hash, outcome) under
   `<storage_root>/audit.log`. Useful as soon as you have more than
   one MCP client.
8. **Evaluation framework.** Pinned-query A/B harness comparing
   provider × retriever combinations on a curated query set. Not
   urgent, but useful once you're choosing models seriously.

## 10. References

Reading list for context, in rough order of "start here":

- [Local-first software: You own your data, in spite of the cloud](https://www.inkandswitch.com/essay/local-first/) — Ink & Switch, 2019
- [Solid Project — About](https://solidproject.org/about) — Tim Berners-Lee
- [Anytype Docs](https://doc.anytype.io/anytype-docs) — working implementation prior art
- [Evergreen notes](https://notes.andymatuschak.org/Evergreen_notes) — Andy Matuschak
- [As We May Think](https://web.mit.edu/sts.035/www/PDFs/think.pdf) — Vannevar Bush, 1945
- [Model Context Protocol](https://modelcontextprotocol.io/) — Anthropic, 2024
- [localfirst.fm podcast](https://www.localfirst.fm/) — practitioner conversations

---

*Last updated: 2026-05-19, after the P0–P8 substrate refactor on
branch `refactor/knowledge-substrate`. Phase history in the plan
file linked above.*
