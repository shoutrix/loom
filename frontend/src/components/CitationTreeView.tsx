import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import {
  X, Loader2, Sparkles, Plus, ExternalLink, ChevronDown, ChevronRight,
} from 'lucide-react'

interface Props {
  paperId: string
  title: string
  onClose: () => void
  onOpenPaper?: (paperId: string, title: string) => void
}

interface NodeSignals {
  paper_id: string
  citation_count: number
  influential_citation_count: number
  local_in_degree: number
  local_pagerank: number
  time_balanced_pagerank: number
  convergence_count: number
  citation_velocity: number
  methodology_ratio: number
  influential_ratio: number
  llm_relevance: number
  score_influence: number
  score_origin: number
  score_frontier: number
  tier: string
  year: number | null
}

interface SubgraphNode {
  paper_id: string
  title: string
  abstract: string
  year: number | null
  venue: string
  citation_count: number
  influential_citation_count: number
  arxiv_id: string
  doi: string
  url: string
  hop_distance: number
}

interface CitationTree {
  version: number
  generated_at: string
  target: { paper_id: string; title: string; year?: number | null; arxiv_id?: string; doi?: string; url?: string }
  subgraph: {
    target_id: string
    nodes: Record<string, SubgraphNode>
    edges: { source: string; target: string; is_influential: boolean; intents: string[] }[]
    stats: Record<string, unknown>
    params: Record<string, unknown>
  } | null
  signals: Record<string, NodeSignals>
  tiers: Record<string, string[]>
  params: Record<string, unknown>
  stats: Record<string, unknown>
}

const TIER_ORDER: { key: string; label: string; description: string }[] = [
  { key: 'origin',      label: 'Origin',      description: 'Seminal papers this line of research traces back to.' },
  { key: 'landmark',    label: 'Landmark',    description: 'High-centrality bridge papers between origin and target.' },
  { key: 'target',      label: 'Target',      description: 'The paper you opened the tree for.' },
  { key: 'convergence', label: 'Convergence', description: 'Papers reached via multiple distinct paths — field-shaping nodes.' },
  { key: 'frontier',    label: 'Frontier',    description: 'Recent high-velocity papers — current state of the art.' },
]

function sourceUrl(node: SubgraphNode | undefined): string {
  if (!node) return ''
  if (node.url) return node.url
  if (node.arxiv_id) return `https://arxiv.org/abs/${node.arxiv_id}`
  if (node.doi) return `https://doi.org/${node.doi}`
  return ''
}

function formatBytes(n: unknown): string {
  if (typeof n !== 'number' || !isFinite(n)) return '—'
  return n.toLocaleString()
}

export function CitationTreeView({ paperId, title, onClose, onOpenPaper }: Props) {
  const [tree, setTree] = useState<CitationTree | null>(null)
  const [loadingMsg, setLoadingMsg] = useState<string>('Looking for cached tree…')
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<string[]>([])
  const [building, setBuilding] = useState(false)
  const [addedIds, setAddedIds] = useState<Set<string>>(new Set())
  const [expandedNode, setExpandedNode] = useState<string | null>(null)
  const pollRef = useRef<number | null>(null)
  const cancelRef = useRef(false)

  useEffect(() => {
    cancelRef.current = false
    void load()
    return () => {
      cancelRef.current = true
      if (pollRef.current !== null) window.clearTimeout(pollRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paperId])

  async function load() {
    setError(null)
    setProgress([])
    setLoadingMsg('Looking for cached tree…')

    try {
      const cached = await api.citationTreeCached(paperId)
      if (cancelRef.current) return
      if (cached?.exists && cached.tree) {
        setTree(cached.tree as CitationTree)
        setLoadingMsg('')
        return
      }
    } catch {
      /* fall through to build */
    }

    await startBuild()
  }

  async function startBuild() {
    setBuilding(true)
    setLoadingMsg('Building citation tree (this may take 30–90 seconds)…')
    try {
      const { job_id } = await api.startCitationTree({
        paper_id: paperId,
        title,
        depth: 3,
        use_llm_tiebreak: true,
      })

      const poll = async () => {
        if (cancelRef.current) return
        const st = await api.citationTreeStatus(job_id)
        const stepsArr = Array.isArray(st?.steps) ? st.steps : []
        const lastFew = stepsArr.slice(-5).map((s: { step: string; info: Record<string, unknown> }) => `${s.step}`)
        setProgress(lastFew)

        if (st.state === 'completed') {
          const cached = await api.citationTreeCached(paperId)
          if (cached?.exists && cached.tree) {
            setTree(cached.tree as CitationTree)
          }
          setBuilding(false)
          setLoadingMsg('')
        } else if (st.state === 'failed') {
          setError(st.error || 'Citation-tree build failed.')
          setBuilding(false)
        } else {
          pollRef.current = window.setTimeout(poll, 2500)
        }
      }
      pollRef.current = window.setTimeout(poll, 2500)
    } catch (e) {
      setError(String(e))
      setBuilding(false)
    }
  }

  async function addToWorkspace(nodePaperId: string) {
    // Reuse the existing queue-by-paper-id endpoint. The id may be in the
    // form "ARXIV:2603.13686" so we pass it as an identifier rather than a
    // paper_id from the registry.
    try {
      await api.queuePapers([], [nodePaperId])
      setAddedIds(prev => new Set(prev).add(nodePaperId))
    } catch {
      // surface inline near the node? For now, silent — the tree view is
      // a read-mostly surface.
    }
  }

  // Render — header + body
  return (
    <div className="fixed inset-0 z-50 bg-white flex flex-col">
      {/* Header */}
      <div className="border-b border-surface-3 px-6 py-3 flex items-center justify-between">
        <div className="flex-1 min-w-0">
          <h2 className="text-sm font-medium text-text-secondary mb-0.5 flex items-center gap-1.5">
            <Sparkles size={13} className="text-accent" /> Citation Tree
          </h2>
          <h1 className="text-base text-text-primary truncate" title={title}>
            {title}
          </h1>
        </div>
        <button
          onClick={onClose}
          className="ml-4 shrink-0 p-1.5 rounded hover:bg-surface-2 text-text-secondary"
          aria-label="Close citation tree"
        >
          <X size={18} />
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto">
        {error && (
          <div className="px-6 py-8 max-w-2xl mx-auto">
            <div className="bg-red-50 text-red-700 text-sm rounded-lg px-4 py-3">
              {error}
              <button
                onClick={() => void startBuild()}
                className="ml-3 underline hover:text-red-900"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        {!error && !tree && (
          <div className="px-6 py-16 max-w-xl mx-auto text-center">
            <Loader2 size={28} className="animate-spin text-text-muted inline-block mb-4" />
            <p className="text-sm text-text-secondary">{loadingMsg}</p>
            {progress.length > 0 && (
              <ul className="mt-4 text-xs text-text-muted font-mono space-y-0.5">
                {progress.map((p, i) => <li key={i}>· {p}</li>)}
              </ul>
            )}
            {building && (
              <p className="mt-4 text-[11px] text-text-muted">
                Walks ~150 references / citations and runs PageRank, convergence
                analysis, and an LLM tie-breaker. Reuses cached results on
                subsequent opens.
              </p>
            )}
          </div>
        )}

        {tree && (
          <div className="max-w-3xl mx-auto px-6 py-6">
            {/* Tree stats */}
            <div className="text-xs text-text-muted mb-4 flex items-center gap-3 flex-wrap">
              <span>{formatBytes((tree.stats as Record<string, unknown>).total_nodes)} nodes</span>
              <span>·</span>
              <span>{formatBytes((tree.stats as Record<string, unknown>).total_edges)} edges</span>
              <span>·</span>
              <span>{formatBytes((tree.stats as Record<string, unknown>).backward_nodes)} backward</span>
              <span>·</span>
              <span>{formatBytes((tree.stats as Record<string, unknown>).forward_nodes)} forward</span>
              <span>·</span>
              <span>built in {formatBytes((tree.stats as Record<string, unknown>).wall_time_total_seconds)}s</span>
              <button
                onClick={() => void startBuild()}
                disabled={building}
                className="ml-auto text-xs text-accent hover:text-accent-dim disabled:opacity-50"
              >
                {building ? 'Rebuilding…' : 'Rebuild'}
              </button>
            </div>

            {/* Tiers */}
            <ol className="space-y-8">
              {TIER_ORDER.map(tierDef => {
                const ids = tree.tiers[tierDef.key] || []
                if (ids.length === 0) {
                  return (
                    <li key={tierDef.key}>
                      <TierHeader label={tierDef.label} description={tierDef.description} count={0} />
                      <p className="text-xs text-text-muted ml-2 mt-1">(none)</p>
                    </li>
                  )
                }
                return (
                  <li key={tierDef.key}>
                    <TierHeader label={tierDef.label} description={tierDef.description} count={ids.length} />
                    <div className="mt-2 space-y-2">
                      {ids.map(pid => {
                        const node = tree.subgraph?.nodes?.[pid]
                        const sig = tree.signals?.[pid]
                        if (!node) return null
                        const expanded = expandedNode === pid
                        const url = sourceUrl(node)
                        const added = addedIds.has(pid)
                        const isTarget = pid === tree.target.paper_id
                        return (
                          <div
                            key={pid}
                            className={`border rounded-lg ${isTarget ? 'border-accent bg-accent/5' : 'border-surface-3 bg-white'}`}
                          >
                            <div className="px-4 py-3">
                              <div className="flex items-start gap-3">
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-baseline gap-2 flex-wrap">
                                    <h3 className="text-sm font-medium text-text-primary leading-snug">
                                      {node.title || '(untitled)'}
                                    </h3>
                                  </div>
                                  <div className="text-xs text-text-muted mt-1 flex items-center gap-2 flex-wrap">
                                    {node.year && <span>{node.year}</span>}
                                    {node.venue && <><span>·</span><span>{node.venue}</span></>}
                                    {typeof node.citation_count === 'number' && (
                                      <><span>·</span><span>{node.citation_count.toLocaleString()} citations</span></>
                                    )}
                                    {url && (
                                      <>
                                        <span>·</span>
                                        <a
                                          href={url}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          className="text-accent hover:underline inline-flex items-center gap-0.5"
                                        >
                                          source <ExternalLink size={10} />
                                        </a>
                                      </>
                                    )}
                                  </div>
                                </div>
                                <div className="shrink-0 flex items-center gap-1">
                                  {!isTarget && (
                                    <button
                                      onClick={() => void addToWorkspace(pid)}
                                      disabled={added}
                                      title={added ? 'Queued' : 'Add to workspace'}
                                      className={`text-xs px-2 py-1 rounded border ${
                                        added
                                          ? 'border-emerald-300 text-emerald-700 bg-emerald-50 cursor-default'
                                          : 'border-accent text-accent hover:bg-accent hover:text-white'
                                      }`}
                                    >
                                      {added ? '✓ Added' : <Plus size={12} className="inline" />}
                                    </button>
                                  )}
                                  {onOpenPaper && (
                                    <button
                                      onClick={() => onOpenPaper(pid, node.title || pid)}
                                      className="text-xs px-2 py-1 rounded border border-surface-3 text-text-secondary hover:bg-surface-2"
                                      title="Open in paper viewer"
                                    >
                                      Open
                                    </button>
                                  )}
                                  <button
                                    onClick={() => setExpandedNode(expanded ? null : pid)}
                                    className="p-1 text-text-muted hover:text-text-primary"
                                    aria-label={expanded ? 'Hide details' : 'Show details'}
                                  >
                                    {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                                  </button>
                                </div>
                              </div>
                              {expanded && sig && (
                                <div className="mt-3 pt-3 border-t border-surface-3 grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
                                  <Sig label="influence" v={sig.score_influence?.toFixed(2)} />
                                  <Sig label="origin score" v={sig.score_origin?.toFixed(2)} />
                                  <Sig label="frontier score" v={sig.score_frontier?.toFixed(2)} />
                                  <Sig label="PageRank" v={sig.local_pagerank?.toFixed(4)} />
                                  <Sig label="time-balanced PR" v={sig.time_balanced_pagerank?.toFixed(2)} />
                                  <Sig label="convergence paths" v={sig.convergence_count} />
                                  <Sig label="local in-degree" v={sig.local_in_degree} />
                                  <Sig label="influential cites (S2)" v={sig.influential_citation_count} />
                                  <Sig label="methodology ratio" v={sig.methodology_ratio?.toFixed(2)} />
                                  <Sig label="citation velocity" v={sig.citation_velocity?.toFixed(1)} />
                                  {sig.llm_relevance > 0 && (
                                    <Sig label="LLM relevance" v={sig.llm_relevance?.toFixed(1)} />
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </li>
                )
              })}
            </ol>
          </div>
        )}
      </div>
    </div>
  )
}

function TierHeader({ label, description, count }: { label: string; description: string; count: number }) {
  return (
    <div>
      <div className="flex items-baseline gap-2">
        <h2 className="text-xs uppercase tracking-wider font-semibold text-text-primary">{label}</h2>
        <span className="text-[10px] text-text-muted">{count}</span>
      </div>
      <p className="text-[11px] text-text-muted mt-0.5">{description}</p>
    </div>
  )
}

function Sig({ label, v }: { label: string; v: string | number | undefined }) {
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-text-muted">{label}</span>
      <span className="font-mono text-text-primary">{v ?? '—'}</span>
    </div>
  )
}

export default CitationTreeView
