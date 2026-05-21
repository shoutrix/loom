import { useState, useEffect, useRef, useCallback } from 'react'
import { api } from '../api/client'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Loader2, FileText, ExternalLink, Plus, CheckCircle, GitBranch, Sparkles,
} from 'lucide-react'

interface Props {
  paperId: string
  title: string
  onOpenPaper?: (paperId: string, title: string) => void
  onOpenCitationTree?: (paperId: string, title: string) => void
}

interface PaperContent {
  content_type: 'pdf_url' | 'markdown'
  url?: string
  pdf_url?: string
  content?: string
  title?: string
  source_url?: string
  error?: string
}

interface GraphPaper {
  id: string
  title: string
  year: number | null
  citation_count: number
  llm_relevance: number
  llm_rationale?: string
  abstract?: string
}

function formatSourceLabel(url: string): string {
  try {
    const u = new URL(url)
    const host = u.hostname.replace(/^www\./, '')
    // Trim trailing path noise, keep up to ~50 chars total.
    const pathBit = u.pathname && u.pathname !== '/'
      ? u.pathname.slice(0, 30) + (u.pathname.length > 30 ? '…' : '')
      : ''
    return host + pathBit
  } catch {
    return url
  }
}

export function PaperViewer({ paperId, title, onOpenPaper, onOpenCitationTree }: Props) {
  const [data, setData] = useState<PaperContent | null>(null)
  const [loading, setLoading] = useState(true)
  const [status, setStatus] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)

  const [exploring, setExploring] = useState(false)
  const [graphPapers, setGraphPapers] = useState<GraphPaper[]>([])
  const [graphError, setGraphError] = useState('')
  const [showGraph, setShowGraph] = useState(false)
  const pollRef = useRef<number | null>(null)

  useEffect(() => {
    setLoading(true)
    setStatus(null)
    setGraphPapers([])
    setShowGraph(false)
    setGraphError('')
    Promise.all([
      api.paperContent(paperId),
      api.registry(),
    ]).then(([content, reg]: [PaperContent, any]) => {
      setData(content)
      const papers = reg.papers || []
      const match = papers.find((p: any) => p.paper_id === paperId)
      if (match) setStatus(match.status)
      setLoading(false)
    }).catch(() => {
      setData({ content_type: 'markdown', content: 'Failed to load paper content.' })
      setLoading(false)
    })
    return () => {
      if (pollRef.current !== null) window.clearTimeout(pollRef.current)
    }
  }, [paperId])

  const addToLibrary = async () => {
    setAdding(true)
    try {
      await api.queuePapers([paperId])
      setStatus('queued')
    } catch {
      // ignore
    }
    setAdding(false)
  }

  const exploreGraph = useCallback(async () => {
    setExploring(true)
    setGraphError('')
    setGraphPapers([])
    setShowGraph(true)
    try {
      const { job_id } = await api.exploreGraph(paperId, title, data?.content?.slice(0, 500) || '')
      const poll = async () => {
        const st = await api.exploreGraphStatus(job_id)
        if (st.state === 'completed') {
          const res = await api.exploreGraphResult(job_id)
          if (res.ready && res.result?.papers) {
            setGraphPapers(res.result.papers)
          }
          setExploring(false)
        } else if (st.state === 'failed') {
          setGraphError(st.error || 'Exploration failed')
          setExploring(false)
        } else {
          pollRef.current = window.setTimeout(poll, 1500)
        }
      }
      pollRef.current = window.setTimeout(poll, 1500)
    } catch {
      setGraphError('Failed to start exploration')
      setExploring(false)
    }
  }, [paperId, title, data])

  const addGraphPaper = async (pid: string) => {
    try {
      await api.queuePapers([pid])
    } catch {
      // ignore
    }
  }

  const isIngested = status === 'ingested'
  const isQueued = status === 'queued' || status === 'ingesting'
  const canAdd = !isIngested && !isQueued

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 size={24} className="animate-spin text-text-muted" />
      </div>
    )
  }

  if (!data || data.error) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-text-muted">
        <FileText size={32} className="mb-2 opacity-30" />
        <p className="text-sm">{data?.error || 'No content available'}</p>
      </div>
    )
  }

  // Compact action strip — kept above the article header so the article
  // itself reads as clean prose with a single prominent source link.
  const actions = (
    <div className="flex items-center justify-end gap-2 px-6 py-2 bg-surface-1 border-b border-surface-3 shrink-0 text-xs">
      {onOpenCitationTree && (
        <button
          onClick={() => onOpenCitationTree(paperId, title)}
          className="flex items-center gap-1.5 px-2.5 py-1 bg-surface-2 text-text-secondary font-medium rounded hover:bg-surface-3 transition-colors"
        >
          <Sparkles size={12} />
          Citation tree
        </button>
      )}
      <button
        onClick={exploreGraph}
        disabled={exploring}
        className="flex items-center gap-1.5 px-2.5 py-1 bg-surface-2 text-text-secondary font-medium rounded hover:bg-surface-3 disabled:opacity-50 transition-colors"
      >
        {exploring ? <Loader2 size={12} className="animate-spin" /> : <GitBranch size={12} />}
        Explore graph
      </button>
      {canAdd && (
        <button
          onClick={addToLibrary}
          disabled={adding}
          className="flex items-center gap-1.5 px-2.5 py-1 bg-accent text-white font-medium rounded hover:bg-accent-dim disabled:opacity-50 transition-colors"
        >
          {adding ? <Loader2 size={12} className="animate-spin" /> : <Plus size={12} />}
          Add to knowledge base
        </button>
      )}
      {isQueued && (
        <span className="flex items-center gap-1 text-amber-700 bg-amber-50 px-2 py-1 rounded">
          <Loader2 size={12} className="animate-spin" /> Queued
        </span>
      )}
      {isIngested && (
        <span className="flex items-center gap-1 text-emerald-700 bg-emerald-50 px-2 py-1 rounded">
          <CheckCircle size={12} /> In knowledge base
        </span>
      )}
    </div>
  )

  const graphPanel = showGraph && (
    <div className="border-b border-surface-3 bg-surface-0 max-h-72 overflow-y-auto">
      <div className="px-6 py-2 border-b border-surface-3 flex items-center justify-between">
        <span className="text-xs font-semibold text-text-secondary">
          {exploring ? 'Exploring citation graph...' : `${graphPapers.length} related papers`}
        </span>
        <button onClick={() => setShowGraph(false)} className="text-xs text-text-muted hover:text-text-primary">
          Close
        </button>
      </div>
      {exploring && (
        <div className="flex items-center justify-center py-6">
          <Loader2 size={18} className="animate-spin text-text-muted mr-2" />
          <span className="text-xs text-text-muted">Traversing citation graph...</span>
        </div>
      )}
      {graphError && (
        <div className="px-6 py-3 text-xs text-red-500">{graphError}</div>
      )}
      {!exploring && graphPapers.length > 0 && (
        <div className="divide-y divide-surface-2">
          {graphPapers.map(p => (
            <div key={p.id} className="px-6 py-2 flex items-start gap-2 hover:bg-surface-1">
              <div className="flex-1 min-w-0">
                <button
                  onClick={() => onOpenPaper?.(p.id, p.title)}
                  className="text-xs font-medium text-text-primary hover:text-accent text-left truncate block w-full"
                  title={p.title}
                >
                  {p.title}
                </button>
                <div className="flex items-center gap-2 mt-0.5">
                  {p.year && <span className="text-[10px] text-text-muted">{p.year}</span>}
                  {p.citation_count > 0 && <span className="text-[10px] text-text-muted">{p.citation_count} cites</span>}
                  <span className="text-[10px] text-text-muted">Score: {p.llm_relevance}/10</span>
                </div>
              </div>
              <button
                onClick={() => addGraphPaper(p.id)}
                className="shrink-0 p-1 text-text-muted hover:text-accent"
                title="Add to library"
              >
                <Plus size={12} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )

  // Shared header: title + prominent source link. Same shape for every paper,
  // whether the body is rendered as markdown or fell back to the iframe path.
  const header = (
    <header className="wiki-header px-6">
      <h1 className="wiki-title">{data.title || title}</h1>
      {data.source_url && (
        <div className="wiki-source">
          Source:{' '}
          <a href={data.source_url} target="_blank" rel="noopener noreferrer">
            {formatSourceLabel(data.source_url)} <ExternalLink size={11} className="inline align-text-bottom" />
          </a>
        </div>
      )}
    </header>
  )

  // Iframe fallback path (no ingested content, only a URL). Rare after P10
  // (shortlisted papers are hidden in SourcesPanel) but still reachable via
  // graph-explore "open" actions and any non-ingested registry entries.
  if (data.content_type === 'pdf_url' && data.url) {
    return (
      <div className="flex flex-col h-full">
        {actions}
        {graphPanel}
        <div className="flex-1 overflow-y-auto">
          <div className="pt-6">{header}</div>
          <div className="px-6 pb-6">
            <a
              href={data.source_url || data.pdf_url || data.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 mt-2 bg-accent text-white text-sm font-medium rounded-lg hover:bg-accent-dim transition-colors"
            >
              Open original in new tab <ExternalLink size={14} />
            </a>
            <p className="text-xs text-text-muted mt-3 max-w-[740px] mx-auto">
              This paper hasn't been ingested into the knowledge base yet, so there's
              no in-app readable view. Click "Add to knowledge base" above to ingest
              it; once ingestion completes, the full text will render here in
              Wikipedia-style markdown.
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {actions}
      {graphPanel}
      <div className="flex-1 overflow-y-auto bg-surface-0">
        <div className="pt-6 pb-12">
          {header}
          <article className="wiki-article px-6">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {data.content || ''}
            </ReactMarkdown>
          </article>
        </div>
      </div>
    </div>
  )
}
