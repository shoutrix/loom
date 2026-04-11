import { useState, useEffect } from 'react'
import { api } from '../api/client'
import {
  Search, Plus, CheckCircle, Clock, AlertCircle, Loader2, BookOpen,
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

interface Props {
  onOpenPaper: (meta: PaperMeta) => void
  onAddSources: () => void
}

const statusIcon: Record<string, React.ReactNode> = {
  shortlisted: <Clock size={12} className="text-text-muted" />,
  queued: <Clock size={12} className="text-amber-500" />,
  ingesting: <Loader2 size={12} className="text-accent animate-spin" />,
  ingested: <CheckCircle size={12} className="text-emerald-500" />,
  failed: <AlertCircle size={12} className="text-red-500" />,
}

export function SourcesPanel({ onOpenPaper, onAddSources }: Props) {
  const [papers, setPapers] = useState<PaperRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')

  const load = async () => {
    try {
      const data = await api.registry()
      setPapers(data.papers || [])
    } catch {
      // ignore
    }
    setLoading(false)
  }

  useEffect(() => {
    load()
    const interval = setInterval(load, 5000)
    return () => clearInterval(interval)
  }, [])

  const filtered = papers.filter(p =>
    !filter || p.title.toLowerCase().includes(filter.toLowerCase())
  )

  const sorted = [...filtered].sort((a, b) => {
    const order: Record<string, number> = { ingesting: 0, queued: 1, ingested: 2, shortlisted: 3, failed: 4 }
    return (order[a.status] ?? 5) - (order[b.status] ?? 5)
  })

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
        <button
          onClick={onAddSources}
          className="w-full flex items-center justify-center gap-1.5 py-2 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-dim transition-colors"
        >
          <Plus size={14} /> Add Sources
        </button>
      </div>

      {/* Paper list */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 size={20} className="animate-spin text-text-muted" />
          </div>
        ) : sorted.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-text-muted px-4">
            <BookOpen size={28} className="mb-2 opacity-30" />
            <p className="text-sm text-center">
              {filter ? 'No matching papers' : 'No papers yet. Click "Add Sources" to get started.'}
            </p>
          </div>
        ) : (
          sorted.map(paper => (
            <button
              key={paper.paper_id}
              onClick={() => onOpenPaper({
                paper_id: paper.paper_id,
                title: paper.title,
                arxiv_id: paper.arxiv_id,
                doi: paper.doi,
              })}
              className="w-full text-left px-3 py-2.5 border-b border-surface-3/50 hover:bg-surface-2 transition-colors group"
            >
              <div className="flex items-start gap-2">
                <div className="mt-0.5 shrink-0">{statusIcon[paper.status] || null}</div>
                <div className="flex-1 min-w-0">
                  <h4 className="text-sm text-text-primary leading-snug line-clamp-2 group-hover:text-accent transition-colors">
                    {paper.title}
                  </h4>
                  <div className="flex items-center gap-2 mt-1 text-xs text-text-muted">
                    <span className="capitalize">{paper.status}</span>
                    {paper.llm_relevance > 0 && (
                      <span className="text-accent">{paper.llm_relevance}/10</span>
                    )}
                  </div>
                  {paper.error && (
                    <p className="text-xs text-red-500 mt-0.5 line-clamp-1">{paper.error}</p>
                  )}
                </div>
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  )
}
