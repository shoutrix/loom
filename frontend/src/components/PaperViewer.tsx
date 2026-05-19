import { useState, useEffect, useRef, useCallback } from 'react'
import { api } from '../api/client'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Loader2, FileText, ExternalLink, Plus, CheckCircle, GitBranch } from 'lucide-react'

interface Props {
  paperId: string
  title: string
  onOpenPaper?: (paperId: string, title: string) => void
}

interface PaperContent {
  content_type: 'pdf_url' | 'markdown'
  url?: string
  pdf_url?: string
  content?: string
  title?: string
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

function IframeFallback({ url, pdfUrl, title }: { url: string; pdfUrl?: string; title: string }) {
  const [failed, setFailed] = useState(false)
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const timerRef = useRef<number | null>(null)

  useEffect(() => {
    setFailed(false)
    // Some sites block iframes silently (no onerror). Detect by checking
    // if the iframe remains blank after a timeout.
    timerRef.current = window.setTimeout(() => {
      try {
        const doc = iframeRef.current?.contentDocument
        // If we can access the document and it has no body content, it likely failed
        if (doc && (!doc.body || doc.body.innerHTML === '')) {
          setFailed(true)
        }
      } catch {
        // Cross-origin — iframe loaded something, which is good
      }
    }, 4000)
    return () => { if (timerRef.current) window.clearTimeout(timerRef.current) }
  }, [url])

  if (failed) {
    const externalUrl = pdfUrl || url
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-4 px-8 text-center">
        <FileText size={48} className="text-text-muted opacity-30" />
        <p className="text-sm text-text-secondary">
          This paper's host does not allow embedded viewing.
        </p>
        <a
          href={externalUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 px-4 py-2 bg-accent text-white text-sm font-medium rounded-lg hover:bg-accent-dim transition-colors"
        >
          Open paper in new tab <ExternalLink size={14} />
        </a>
      </div>
    )
  }

  return (
    <iframe
      ref={iframeRef}
      src={url}
      className="flex-1 w-full border-0"
      title={title}
      onError={() => setFailed(true)}
    />
  )
}

export function PaperViewer({ paperId, title, onOpenPaper }: Props) {
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

  const toolbar = (
    <div className="flex items-center justify-between px-4 py-2 bg-surface-1 border-b border-surface-3 shrink-0">
      <span className="text-sm text-text-secondary truncate flex-1 mr-3">{title}</span>
      <div className="flex items-center gap-2 shrink-0">
        <button
          onClick={exploreGraph}
          disabled={exploring}
          className="flex items-center gap-1.5 px-3 py-1 bg-surface-2 text-text-secondary text-xs font-medium rounded-lg hover:bg-surface-3 disabled:opacity-50 transition-colors"
        >
          {exploring ? <Loader2 size={12} className="animate-spin" /> : <GitBranch size={12} />}
          Explore Graph
        </button>
        {canAdd && (
          <button
            onClick={addToLibrary}
            disabled={adding}
            className="flex items-center gap-1.5 px-3 py-1 bg-accent text-white text-xs font-medium rounded-lg hover:bg-accent-dim disabled:opacity-50 transition-colors"
          >
            {adding ? <Loader2 size={12} className="animate-spin" /> : <Plus size={12} />}
            Add to Knowledge Base
          </button>
        )}
        {isQueued && (
          <span className="flex items-center gap-1 text-xs text-amber-600 bg-amber-50 px-2 py-1 rounded-lg">
            <Loader2 size={12} className="animate-spin" /> Queued
          </span>
        )}
        {isIngested && (
          <span className="flex items-center gap-1 text-xs text-emerald-600 bg-emerald-50 px-2 py-1 rounded-lg">
            <CheckCircle size={12} /> In Knowledge Base
          </span>
        )}
        {data.content_type === 'pdf_url' && data.url && (
          <a
            href={data.pdf_url || data.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs text-accent hover:text-accent-dim"
          >
            Open original <ExternalLink size={12} />
          </a>
        )}
      </div>
    </div>
  )

  const graphPanel = showGraph && (
    <div className="border-t border-surface-3 bg-surface-0 max-h-72 overflow-y-auto">
      <div className="px-4 py-2 border-b border-surface-3 flex items-center justify-between">
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
        <div className="px-4 py-3 text-xs text-red-500">{graphError}</div>
      )}
      {!exploring && graphPapers.length > 0 && (
        <div className="divide-y divide-surface-2">
          {graphPapers.map(p => (
            <div key={p.id} className="px-4 py-2 flex items-start gap-2 hover:bg-surface-1">
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

  if (data.content_type === 'pdf_url' && data.url) {
    return (
      <div className="flex flex-col h-full">
        {toolbar}
        {graphPanel}
        <IframeFallback url={data.url} pdfUrl={data.pdf_url} title={title} />
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {toolbar}
      {graphPanel}
      <div className="flex-1 overflow-y-auto px-8 py-6">
        <div className="prose max-w-3xl mx-auto text-sm">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{data.content || ''}</ReactMarkdown>
        </div>
      </div>
    </div>
  )
}
