import { useState, useEffect, useRef, useMemo } from 'react'
import { api } from '../api/client'
import {
  Search, Plus, CheckCircle, Clock, AlertCircle, Loader2, BookOpen,
  ChevronRight, ChevronDown, Sparkles, RefreshCw,
} from 'lucide-react'

interface PaperRecord {
  paper_id: string
  title: string
  status: string
  arxiv_id: string
  doi: string
  llm_relevance: number
  ingested_at: string
  error: string
}

interface PaperMeta {
  paper_id: string
  title: string
  arxiv_id?: string
  doi?: string
}

interface Group {
  name: string
  description?: string
  paper_ids?: string[]
  subgroups?: Group[]
}

interface CategorizationPayload {
  current_paper_count: number
  min_papers_to_categorize: number
  stale: boolean
  exists: boolean
  categorization?: {
    version: number
    generated_at: string
    model: string
    paper_count: number
    summaries: Record<string, string>
    hierarchy: Group[]
  }
}

interface Props {
  onOpenPaper: (meta: PaperMeta) => void
  onAddSources: () => void
}

const statusIcon: Record<string, React.ReactNode> = {
  queued: <Clock size={12} className="text-amber-500" />,
  ingesting: <Loader2 size={12} className="text-accent animate-spin" />,
  ingested: <CheckCircle size={12} className="text-emerald-500" />,
  failed: <AlertCircle size={12} className="text-red-500" />,
}

export function SourcesPanel({ onOpenPaper, onAddSources }: Props) {
  const [papers, setPapers] = useState<PaperRecord[]>([])
  const [cat, setCat] = useState<CategorizationPayload | null>(null)
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')
  const [categorizing, setCategorizing] = useState(false)
  const [categorizeError, setCategorizeError] = useState<string | null>(null)
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})
  const autoKickRef = useRef(false)
  const pollRef = useRef<number | null>(null)

  const load = async () => {
    try {
      const [reg, c] = await Promise.all([
        api.registry(),
        api.categorization(),
      ])
      setPapers(reg.papers || [])
      setCat(c)
    } catch {
      // ignore
    }
    setLoading(false)
  }

  useEffect(() => {
    load()
    const interval = setInterval(load, 5000)
    return () => {
      clearInterval(interval)
      if (pollRef.current !== null) window.clearTimeout(pollRef.current)
    }
  }, [])

  // Auto-kick categorization once per workspace if missing/stale and there
  // are enough papers. autoKickRef guards against firing repeatedly.
  useEffect(() => {
    if (!cat || autoKickRef.current || categorizing) return
    const enough = cat.current_paper_count >= cat.min_papers_to_categorize
    if (!enough) return
    if (!cat.exists || cat.stale) {
      autoKickRef.current = true
      void runCategorize()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cat, categorizing])

  const runCategorize = async () => {
    setCategorizing(true)
    setCategorizeError(null)
    try {
      const { job_id } = await api.startCategorize()
      const poll = async () => {
        try {
          const st = await api.categorizeStatus(job_id)
          if (st.state === 'completed') {
            setCategorizing(false)
            void load()
          } else if (st.state === 'failed') {
            setCategorizeError(st.error || 'Categorization failed')
            setCategorizing(false)
          } else {
            pollRef.current = window.setTimeout(poll, 2000)
          }
        } catch {
          setCategorizeError('Status check failed')
          setCategorizing(false)
        }
      }
      pollRef.current = window.setTimeout(poll, 2000)
    } catch {
      setCategorizeError('Failed to start categorization')
      setCategorizing(false)
    }
  }

  // Hide search-found-but-never-added papers (P10).
  const added = papers.filter(p => p.status !== 'shortlisted')

  const papersById = useMemo<Record<string, PaperRecord>>(() => {
    const m: Record<string, PaperRecord> = {}
    for (const p of added) m[p.paper_id] = p
    return m
  }, [added])

  const summaries = cat?.categorization?.summaries || {}
  const hierarchy = cat?.categorization?.hierarchy || []

  const filterLower = filter.trim().toLowerCase()
  const matchesFilter = (paper: PaperRecord) =>
    !filterLower ||
    paper.title.toLowerCase().includes(filterLower) ||
    (summaries[paper.paper_id] || '').toLowerCase().includes(filterLower)

  // For the flat fallback (no categorization or below-min), sort like before.
  const flatSorted = useMemo(() => {
    const order: Record<string, number> = { ingesting: 0, queued: 1, ingested: 2, failed: 3 }
    return [...added]
      .filter(matchesFilter)
      .sort((a, b) => (order[a.status] ?? 4) - (order[b.status] ?? 4))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [added, filterLower, summaries])

  const renderPaperRow = (paper: PaperRecord) => {
    const summary = summaries[paper.paper_id]
    return (
      <button
        key={paper.paper_id}
        onClick={() =>
          onOpenPaper({
            paper_id: paper.paper_id,
            title: paper.title,
            arxiv_id: paper.arxiv_id,
            doi: paper.doi,
          })
        }
        className="w-full text-left px-3 py-2 border-b border-surface-3/50 hover:bg-surface-2 transition-colors group"
      >
        <div className="flex items-start gap-2">
          <div className="mt-0.5 shrink-0">{statusIcon[paper.status] || null}</div>
          <div className="flex-1 min-w-0">
            <h4 className="text-sm text-text-primary leading-snug line-clamp-2 group-hover:text-accent transition-colors">
              {paper.title}
            </h4>
            {summary && (
              <p className="text-xs text-text-secondary mt-0.5 leading-snug line-clamp-2">
                {summary}
              </p>
            )}
            {paper.error && (
              <p className="text-xs text-red-500 mt-0.5 line-clamp-1">{paper.error}</p>
            )}
          </div>
        </div>
      </button>
    )
  }

  const renderGroup = (group: Group, depth: number, pathKey: string): React.ReactNode => {
    const groupPaperIds = (group.paper_ids || []).filter(id => papersById[id])
    const subgroups = group.subgroups || []
    const visiblePapers = groupPaperIds
      .map(id => papersById[id])
      .filter(matchesFilter)
    const visibleSubgroups = subgroups
      .map(sg => renderGroup(sg, depth + 1, pathKey + '/' + sg.name))
      .filter(node => node !== null)

    // If filter is active and this group has nothing to show, hide it entirely.
    if (filterLower && visiblePapers.length === 0 && visibleSubgroups.length === 0) {
      return null
    }

    // Auto-expand when a filter is active; otherwise honor user state.
    const expanded = filterLower ? true : !collapsed[pathKey]
    const totalCount = groupPaperIds.length + subgroups.reduce(
      (acc, sg) => acc + (sg.paper_ids?.length || 0), 0,
    )

    return (
      <div key={pathKey} className={depth === 0 ? '' : 'ml-3'}>
        <button
          onClick={() => setCollapsed(c => ({ ...c, [pathKey]: !c[pathKey] }))}
          className="w-full flex items-start gap-1 px-2 py-1.5 hover:bg-surface-1 text-left"
        >
          <div className="mt-0.5 shrink-0 text-text-muted">
            {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-baseline gap-2">
              <span className={`text-xs uppercase tracking-wide font-semibold ${depth === 0 ? 'text-text-primary' : 'text-text-secondary'}`}>
                {group.name}
              </span>
              <span className="text-[10px] text-text-muted shrink-0">
                {totalCount}
              </span>
            </div>
            {group.description && (
              <p className="text-[11px] text-text-muted leading-snug mt-0.5">
                {group.description}
              </p>
            )}
          </div>
        </button>
        {expanded && (
          <div className="border-l border-surface-3 ml-3">
            {visiblePapers.map(renderPaperRow)}
            {visibleSubgroups}
          </div>
        )}
      </div>
    )
  }

  const hasCategorization = hierarchy.length > 0
  const totalAdded = added.length
  const shouldShowTree =
    hasCategorization && totalAdded >= (cat?.min_papers_to_categorize ?? 3)

  const generatedAt = cat?.categorization?.generated_at
  const generatedAtLocal = generatedAt
    ? new Date(generatedAt).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    })
    : ''

  return (
    <div className="flex flex-col h-full">
      {/* Top bar */}
      <div className="p-3 space-y-2">
        <div className="flex items-center gap-2">
          <div className="flex-1 relative">
            <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
            <input
              className="w-full bg-white border border-surface-3 rounded-lg pl-8 pr-3 py-1.5 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-accent"
              placeholder="Filter papers..."
              value={filter}
              onChange={e => setFilter(e.target.value)}
            />
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onAddSources}
            className="flex-1 flex items-center justify-center gap-1.5 py-2 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-dim transition-colors"
          >
            <Plus size={14} /> Add Sources
          </button>
          <button
            onClick={runCategorize}
            disabled={categorizing || totalAdded < (cat?.min_papers_to_categorize ?? 3)}
            title={
              totalAdded < (cat?.min_papers_to_categorize ?? 3)
                ? `Need at least ${cat?.min_papers_to_categorize ?? 3} added papers`
                : 'Regenerate categorization'
            }
            className="flex items-center justify-center gap-1.5 px-3 py-2 bg-surface-2 text-text-secondary rounded-lg text-sm font-medium hover:bg-surface-3 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {categorizing ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <RefreshCw size={14} />
            )}
          </button>
        </div>
        {categorizing && (
          <div className="flex items-center gap-1.5 px-2 py-1 text-xs text-text-muted">
            <Sparkles size={11} className="text-accent" />
            <span>Organizing papers…</span>
          </div>
        )}
        {!categorizing && cat?.stale && hasCategorization && (
          <div className="text-[11px] text-amber-700 bg-amber-50 px-2 py-1 rounded">
            Paper count drifted — regenerate categorization for an up-to-date view.
          </div>
        )}
        {categorizeError && (
          <div className="text-[11px] text-red-600 bg-red-50 px-2 py-1 rounded">
            {categorizeError}
          </div>
        )}
        {generatedAtLocal && (
          <div className="text-[10px] text-text-muted px-2">
            Categorization · {generatedAtLocal} · {cat?.categorization?.model}
          </div>
        )}
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 size={20} className="animate-spin text-text-muted" />
          </div>
        ) : totalAdded === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-text-muted px-4">
            <BookOpen size={28} className="mb-2 opacity-30" />
            <p className="text-sm text-center">
              {filter ? 'No matching papers' : 'No papers yet. Click "Add Sources" to get started.'}
            </p>
          </div>
        ) : shouldShowTree ? (
          <div className="py-1">
            {hierarchy.map((g, i) => renderGroup(g, 0, `g${i}/${g.name}`))}
          </div>
        ) : (
          // Flat fallback: too few papers to categorize, or hierarchy empty.
          flatSorted.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-text-muted px-4">
              <BookOpen size={28} className="mb-2 opacity-30" />
              <p className="text-sm text-center">No matching papers</p>
            </div>
          ) : (
            flatSorted.map(renderPaperRow)
          )
        )}
      </div>
    </div>
  )
}
