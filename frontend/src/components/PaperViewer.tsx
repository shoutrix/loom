import { useState, useEffect } from 'react'
import { api } from '../api/client'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Loader2, FileText, ExternalLink, Plus, CheckCircle } from 'lucide-react'

interface Props {
  paperId: string
  title: string
}

interface PaperContent {
  content_type: 'pdf_url' | 'markdown'
  url?: string
  content?: string
  title?: string
  error?: string
}

export function PaperViewer({ paperId, title }: Props) {
  const [data, setData] = useState<PaperContent | null>(null)
  const [loading, setLoading] = useState(true)
  const [status, setStatus] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)

  useEffect(() => {
    setLoading(true)
    setStatus(null)
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
            href={data.url}
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

  if (data.content_type === 'pdf_url' && data.url) {
    return (
      <div className="flex flex-col h-full">
        {toolbar}
        <iframe src={data.url} className="flex-1 w-full border-0" title={title} />
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {toolbar}
      <div className="flex-1 overflow-y-auto px-8 py-6">
        <div className="prose max-w-3xl mx-auto text-sm">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{data.content || ''}</ReactMarkdown>
        </div>
      </div>
    </div>
  )
}
