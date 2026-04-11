import { useState, useRef } from 'react'
import { api } from '../api/client'
import {
  X, Search, Loader2, ChevronDown, ChevronUp, Zap, Link, Upload, Plus,
} from 'lucide-react'

interface Paper {
  id: string
  title: string
  abstract: string
  year: number | null
  citation_count: number
  arxiv_id: string
  doi: string
  llm_relevance: number
  llm_rationale: string
  importance_score: number
  is_survey: boolean
  venue: string
}

interface SearchResult {
  papers: Paper[]
  stats: {
    papers_retrieved: number
    papers_after_dedup: number
    papers_after_llm_filter: number
    papers_returned: number
    total_seconds: number
  }
}

interface Props {
  onClose: () => void
}

export function AddSourcesModal({ onClose }: Props) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [expanded, setExpanded] = useState<string | null>(null)
  const [queueing, setQueueing] = useState(false)

  const [urlInput, setUrlInput] = useState('')
  const [urlLoading, setUrlLoading] = useState(false)
  const [urlMessage, setUrlMessage] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  const [fileLoading, setFileLoading] = useState(false)
  const [fileMessage, setFileMessage] = useState('')

  const [mode, setMode] = useState<'search' | 'manual'>('search')
  const [maxResults, setMaxResults] = useState(20)

  const doSearch = async () => {
    if (!query.trim() || loading) return
    setLoading(true)
    setResults(null)
    setSelected(new Set())
    try {
      const data = await api.searchPapers(query, maxResults)
      setResults(data)
    } catch {
      // ignore
    }
    setLoading(false)
  }

  const toggleSelect = (id: string) => {
    setSelected(prev => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  const selectAll = () => {
    if (!results) return
    if (selected.size === results.papers.length) {
      setSelected(new Set())
    } else {
      setSelected(new Set(results.papers.map(p => p.id)))
    }
  }

  const queueSelected = async () => {
    if (selected.size === 0) return
    setQueueing(true)
    try {
      await api.queuePapers(Array.from(selected))
      setSelected(new Set())
    } catch {
      // ignore
    }
    setQueueing(false)
  }

  const ingestUrl = async () => {
    if (!urlInput.trim()) return
    setUrlLoading(true)
    setUrlMessage('')
    try {
      const data = await api.ingestUrl(urlInput.trim())
      setUrlMessage(data.error ? `Error: ${data.error}` : `Ingested: ${data.title || 'done'}`)
      setUrlInput('')
    } catch {
      setUrlMessage('Failed to ingest URL')
    }
    setUrlLoading(false)
  }

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setFileLoading(true)
    setFileMessage('')
    try {
      const data = await api.ingestFile(file)
      setFileMessage(data.error ? `Error: ${data.error}` : `Ingested: ${data.title || file.name}`)
    } catch {
      setFileMessage('Failed to ingest file')
    }
    setFileLoading(false)
    if (fileRef.current) fileRef.current.value = ''
  }

  const relevanceColor = (score: number) => {
    if (score >= 9) return 'text-emerald-600'
    if (score >= 7) return 'text-blue-600'
    if (score >= 5) return 'text-amber-600'
    return 'text-text-muted'
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-12 bg-black/30 backdrop-blur-sm"
         onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="w-full max-w-3xl max-h-[85vh] bg-white rounded-xl shadow-2xl border border-surface-3 flex flex-col overflow-hidden">

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-surface-3">
          <h2 className="text-lg font-semibold text-text-primary">Add Sources</h2>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary transition-colors">
            <X size={20} />
          </button>
        </div>

        {/* Mode toggle */}
        <div className="flex gap-1 px-5 pt-4">
          <button
            onClick={() => setMode('search')}
            className={`px-3 py-1.5 text-sm rounded-lg font-medium transition-colors ${
              mode === 'search' ? 'bg-accent text-white' : 'text-text-secondary hover:bg-surface-2'
            }`}
          >
            <Search size={13} className="inline mr-1.5 -mt-0.5" />
            Search Papers
          </button>
          <button
            onClick={() => setMode('manual')}
            className={`px-3 py-1.5 text-sm rounded-lg font-medium transition-colors ${
              mode === 'manual' ? 'bg-accent text-white' : 'text-text-secondary hover:bg-surface-2'
            }`}
          >
            <Plus size={13} className="inline mr-1.5 -mt-0.5" />
            Add Manually
          </button>
        </div>

        {mode === 'search' ? (
          <>
            {/* Search bar */}
            <div className="px-5 py-4">
              <div className="flex gap-2">
                <div className="flex-1 relative">
                  <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
                  <input
                    className="w-full bg-surface-1 border border-surface-3 rounded-xl pl-10 pr-4 py-3 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent/20"
                    placeholder="Search for papers across arXiv, Semantic Scholar, OpenAlex..."
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && doSearch()}
                    autoFocus
                  />
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  <input
                    type="number"
                    min={5}
                    max={50}
                    value={maxResults}
                    onChange={e => setMaxResults(Math.max(5, Math.min(50, Number(e.target.value) || 20)))}
                    className="w-14 bg-surface-1 border border-surface-3 rounded-lg px-2 py-3 text-sm text-center text-text-primary focus:outline-none focus:border-accent"
                    title="Number of papers to return"
                  />
                  <span className="text-xs text-text-muted whitespace-nowrap">papers</span>
                </div>
                <button
                  onClick={doSearch}
                  disabled={loading || !query.trim()}
                  className="px-5 py-3 bg-accent text-white rounded-xl text-sm font-medium hover:bg-accent-dim disabled:opacity-40 transition-colors"
                >
                  {loading ? <Loader2 size={16} className="animate-spin" /> : 'Search'}
                </button>
              </div>
            </div>

            {/* Results */}
            <div className="flex-1 overflow-y-auto px-5 pb-4">
              {loading && (
                <div className="flex flex-col items-center justify-center py-12 text-text-secondary">
                  <Loader2 size={28} className="animate-spin mb-3" />
                  <span className="text-sm">Searching across sources...</span>
                  <span className="text-xs text-text-muted mt-1">This may take 30-60 seconds</span>
                </div>
              )}

              {results && !loading && (
                <>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs text-text-muted">
                      {results.stats.papers_returned} papers in {results.stats.total_seconds}s
                    </span>
                    <div className="flex gap-2 items-center">
                      <button onClick={selectAll} className="text-xs text-accent hover:text-accent-dim">
                        {selected.size === results.papers.length ? 'Deselect all' : 'Select all'}
                      </button>
                      {selected.size > 0 && (
                        <button
                          onClick={queueSelected}
                          disabled={queueing}
                          className="bg-accent text-white px-3 py-1 rounded-lg text-xs font-medium hover:bg-accent-dim disabled:opacity-50"
                        >
                          {queueing ? 'Adding...' : `Add ${selected.size} to Library`}
                        </button>
                      )}
                    </div>
                  </div>

                  <div className="space-y-1">
                    {results.papers.map(paper => (
                      <div
                        key={paper.id}
                        className={`rounded-lg border transition-colors ${
                          selected.has(paper.id)
                            ? 'bg-accent/5 border-accent/20'
                            : 'bg-white border-surface-3 hover:border-surface-3'
                        }`}
                      >
                        <div className="px-3 py-2.5 flex items-start gap-3">
                          <input
                            type="checkbox"
                            checked={selected.has(paper.id)}
                            onChange={() => toggleSelect(paper.id)}
                            className="mt-1 accent-accent"
                          />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-0.5">
                              <span className={`text-xs font-mono font-bold ${relevanceColor(paper.llm_relevance)}`}>
                                {paper.llm_relevance}/10
                              </span>
                              {paper.is_survey && (
                                <span className="text-[10px] px-1.5 py-0.5 bg-purple-100 text-purple-700 rounded font-medium">
                                  SURVEY
                                </span>
                              )}
                            </div>
                            <h4 className="text-sm font-medium text-text-primary leading-snug line-clamp-2">
                              {paper.title}
                            </h4>
                            <div className="flex items-center gap-2 mt-1 text-xs text-text-muted">
                              {paper.year && <span>{paper.year}</span>}
                              {paper.citation_count > 0 && (
                                <span className="flex items-center gap-0.5">
                                  <Zap size={10} /> {paper.citation_count}
                                </span>
                              )}
                              {paper.venue && <span className="truncate max-w-[180px]">{paper.venue}</span>}
                            </div>
                            {expanded === paper.id && (
                              <div className="mt-2 space-y-1.5">
                                <p className="text-xs text-accent italic">"{paper.llm_rationale}"</p>
                                {paper.abstract && (
                                  <p className="text-xs text-text-secondary line-clamp-4">{paper.abstract}</p>
                                )}
                              </div>
                            )}
                          </div>
                          <button
                            onClick={() => setExpanded(expanded === paper.id ? null : paper.id)}
                            className="text-text-muted hover:text-text-secondary mt-0.5 shrink-0"
                          >
                            {expanded === paper.id ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {!results && !loading && (
                <div className="flex flex-col items-center justify-center py-16 text-text-muted">
                  <Search size={36} className="mb-3 opacity-20" />
                  <p className="text-sm">Search across arXiv, Semantic Scholar, OpenAlex, and Crossref</p>
                </div>
              )}
            </div>
          </>
        ) : (
          /* Manual add: URL or file */
          <div className="px-5 py-6 space-y-6">
            {/* URL */}
            <div>
              <label className="text-sm font-medium text-text-primary mb-2 flex items-center gap-1.5">
                <Link size={14} /> Add by URL
              </label>
              <div className="flex gap-2 mt-2">
                <input
                  className="flex-1 bg-surface-1 border border-surface-3 rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-accent"
                  placeholder="Paste arXiv URL, DOI, or any web page..."
                  value={urlInput}
                  onChange={e => setUrlInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && ingestUrl()}
                />
                <button
                  onClick={ingestUrl}
                  disabled={urlLoading || !urlInput.trim()}
                  className="px-4 py-2 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-dim disabled:opacity-40 transition-colors"
                >
                  {urlLoading ? <Loader2 size={14} className="animate-spin" /> : 'Add'}
                </button>
              </div>
              {urlMessage && (
                <p className={`text-xs mt-2 ${urlMessage.startsWith('Error') ? 'text-red-500' : 'text-emerald-600'}`}>
                  {urlMessage}
                </p>
              )}
            </div>

            {/* File upload */}
            <div>
              <label className="text-sm font-medium text-text-primary mb-2 flex items-center gap-1.5">
                <Upload size={14} /> Upload a file
              </label>
              <div className="mt-2">
                <input
                  ref={fileRef}
                  type="file"
                  accept=".pdf,.md,.txt,.docx,.pptx"
                  onChange={handleFile}
                  className="hidden"
                />
                <button
                  onClick={() => fileRef.current?.click()}
                  disabled={fileLoading}
                  className="w-full border-2 border-dashed border-surface-3 rounded-lg py-8 text-sm text-text-muted hover:border-accent hover:text-accent transition-colors"
                >
                  {fileLoading ? (
                    <Loader2 size={20} className="animate-spin mx-auto" />
                  ) : (
                    <>Click to select a file (PDF, Markdown, DOCX, PPTX)</>
                  )}
                </button>
              </div>
              {fileMessage && (
                <p className={`text-xs mt-2 ${fileMessage.startsWith('Error') ? 'text-red-500' : 'text-emerald-600'}`}>
                  {fileMessage}
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
