import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from '../api/client'
import {
  Loader2, ExternalLink, RefreshCw, ChevronDown, ChevronRight, FileText,
} from 'lucide-react'

interface Dataset {
  name: string
  size: string
  type: string
}

interface RelatedWork {
  title: string
  why: string
}

export interface PaperCardData {
  version: number
  generated_at: string
  model: string
  paper_id: string

  title: string
  authors: string[]
  venue: string
  year: number | null
  source_url: string
  arxiv_id: string
  doi: string

  tldr: string
  problem: string
  approach: string
  contributions: string[]
  datasets: Dataset[]
  setup: string
  results: string[]
  conclusion: string
  strengths: string[]
  limitations: string[]
  related_work: RelatedWork[]
  workspace_relevance: string
  open_questions: string[]
}

interface Props {
  paperId: string
  title: string
  /** Full paper markdown (frontmatter-stripped). Shown when "View full text" is expanded. */
  fullMarkdown?: string
  /** Canonical source URL surfaced at the top of the card. */
  sourceUrl?: string
}

function formatSourceLabel(url: string): string {
  try {
    const u = new URL(url)
    return u.hostname.replace(/^www\./, '') + (u.pathname && u.pathname !== '/' ? u.pathname.slice(0, 32) : '')
  } catch {
    return url
  }
}

function generatedAtLocal(iso: string): string {
  if (!iso) return ''
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    })
  } catch {
    return iso
  }
}

export function PaperCard({ paperId, title, fullMarkdown, sourceUrl }: Props) {
  const [card, setCard] = useState<PaperCardData | null>(null)
  const [loadingMsg, setLoadingMsg] = useState<string>('Looking for cached card…')
  const [error, setError] = useState<string | null>(null)
  const [building, setBuilding] = useState(false)
  const [showFullText, setShowFullText] = useState(false)
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
    setLoadingMsg('Looking for cached card…')
    try {
      const cached = await api.paperCardCached(paperId)
      if (cancelRef.current) return
      if (cached?.exists && cached.card) {
        setCard(cached.card as PaperCardData)
        setLoadingMsg('')
        return
      }
    } catch {
      /* fall through */
    }
    await startBuild()
  }

  async function startBuild() {
    setBuilding(true)
    setLoadingMsg('Extracting structured review (~10–20 seconds)…')
    try {
      const { job_id } = await api.startPaperCard(paperId)
      const poll = async () => {
        if (cancelRef.current) return
        const st = await api.paperCardStatus(job_id)
        if (st.state === 'completed') {
          const cached = await api.paperCardCached(paperId)
          if (cached?.exists && cached.card) {
            setCard(cached.card as PaperCardData)
          }
          setBuilding(false)
          setLoadingMsg('')
        } else if (st.state === 'failed') {
          setError(st.error || 'Card extraction failed.')
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

  // ---- render ----

  if (loadingMsg || building) {
    return (
      <div className="px-6 py-16 max-w-2xl mx-auto text-center">
        <Loader2 size={24} className="animate-spin text-text-muted inline-block mb-3" />
        <p className="text-sm text-text-secondary">{loadingMsg || 'Building…'}</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="px-6 py-8 max-w-2xl mx-auto">
        <div className="bg-red-50 text-red-700 text-sm rounded-lg px-4 py-3">
          {error}
          <button onClick={() => void startBuild()} className="ml-3 underline hover:text-red-900">
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!card) {
    return (
      <div className="px-6 py-16 max-w-xl mx-auto text-center text-text-muted">
        <FileText size={28} className="mx-auto mb-3 opacity-40" />
        <p className="text-sm">No card available for this paper.</p>
      </div>
    )
  }

  const url = card.source_url || sourceUrl
  const titleText = card.title || title
  const generated = generatedAtLocal(card.generated_at)

  return (
    <div className="paper-card-container px-6 py-6 max-w-3xl mx-auto">
      {/* Header */}
      <header className="paper-card-header mb-5">
        <h1 className="text-2xl font-medium text-text-primary mb-1 leading-tight">
          {titleText}
        </h1>
        <div className="text-sm text-text-secondary mb-2">
          {card.authors.length > 0 && (
            <span>{card.authors.slice(0, 6).join(', ')}{card.authors.length > 6 ? ' et al.' : ''}</span>
          )}
          {(card.venue || card.year) && (
            <>
              {card.authors.length > 0 && <span className="mx-2">·</span>}
              {card.venue && <span className="meta-pill mr-1.5">{card.venue}</span>}
              {card.year !== null && <span className="meta-pill">{card.year}</span>}
            </>
          )}
        </div>
        {url && (
          <div className="text-xs text-text-muted">
            Source:{' '}
            <a href={url} target="_blank" rel="noopener noreferrer" className="text-accent hover:underline inline-flex items-center gap-0.5">
              {formatSourceLabel(url)} <ExternalLink size={10} />
            </a>
          </div>
        )}
        <div className="mt-2 flex items-center gap-3 text-[11px] text-text-muted">
          {generated && <span>Card · {generated} · {card.model}</span>}
          <button
            onClick={() => void startBuild()}
            className="ml-auto inline-flex items-center gap-1 text-accent hover:text-accent-dim"
            title="Regenerate card"
          >
            <RefreshCw size={11} /> Regenerate
          </button>
        </div>
      </header>

      {/* TL;DR */}
      {card.tldr && (
        <div className="paper-card-tldr mb-5">
          {card.tldr}
        </div>
      )}

      {/* Three primary "field" rows */}
      <Field label="Problem" body={card.problem} />
      <Field label="Approach" body={card.approach} />
      <FieldList label="Contributions" items={card.contributions} />

      {/* Datasets */}
      {card.datasets.length > 0 && (
        <div className="paper-card-section">
          <h2 className="paper-card-section-label">Datasets</h2>
          <div className="flex flex-wrap gap-2 mt-1.5">
            {card.datasets.map((d, i) => (
              <div key={i} className="dataset-chip">
                <span className="dataset-name">{d.name || '(unnamed)'}</span>
                {(d.size || d.type) && (
                  <span className="dataset-meta">
                    {[d.size, d.type].filter(Boolean).join(' · ')}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <Field label="Setup" body={card.setup} />
      <FieldList label="Headline results" items={card.results} resultStyle />
      <Field label="Conclusion" body={card.conclusion} />

      {/* Strengths / Limitations — two-column grid in VoxSpar idiom */}
      {(card.strengths.length > 0 || card.limitations.length > 0) && (
        <div className="paper-card-section">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-1.5">
            {card.strengths.length > 0 && (
              <div className="gb-block gb-good">
                <h4>Strengths</h4>
                <ul>
                  {card.strengths.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </div>
            )}
            {card.limitations.length > 0 && (
              <div className="gb-block gb-bad">
                <h4>Limitations</h4>
                <ul>
                  {card.limitations.map((l, i) => <li key={i}>{l}</li>)}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Related work */}
      {card.related_work.length > 0 && (
        <div className="paper-card-section">
          <h2 className="paper-card-section-label">Related work cited</h2>
          <ul className="paper-card-related mt-1.5">
            {card.related_work.map((r, i) => (
              <li key={i}>
                <span className="related-title">{r.title}</span>
                {r.why && <span className="related-why"> — {r.why}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Workspace relevance */}
      {card.workspace_relevance && (
        <div className="paper-card-relevance">
          <h4>Workspace relevance</h4>
          <p>{card.workspace_relevance}</p>
        </div>
      )}

      {/* Open questions */}
      {card.open_questions.length > 0 && (
        <div className="paper-card-section">
          <h2 className="paper-card-section-label">Open questions</h2>
          <ul className="paper-card-list mt-1.5">
            {card.open_questions.map((q, i) => <li key={i}>{q}</li>)}
          </ul>
        </div>
      )}

      {/* View full text expander */}
      {fullMarkdown && (
        <div className="mt-8 pt-5 border-t border-surface-3">
          <button
            onClick={() => setShowFullText(!showFullText)}
            className="text-sm text-text-secondary hover:text-text-primary inline-flex items-center gap-1.5"
          >
            {showFullText ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            View full text ({fullMarkdown.length.toLocaleString()} chars)
          </button>
          {showFullText && (
            <div className="mt-4">
              <article className="wiki-article">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {fullMarkdown}
                </ReactMarkdown>
              </article>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ---- small render helpers ----

function Field({ label, body }: { label: string; body: string }) {
  if (!body) return null
  return (
    <div className="paper-card-field">
      <span className="paper-card-field-label">{label}:</span>
      <span className="paper-card-field-body"> {body}</span>
    </div>
  )
}

function FieldList({
  label,
  items,
  resultStyle = false,
}: {
  label: string
  items: string[]
  resultStyle?: boolean
}) {
  if (!items || items.length === 0) return null
  return (
    <div className="paper-card-section">
      <h2 className="paper-card-section-label">{label}</h2>
      <ul className={resultStyle ? 'paper-card-results' : 'paper-card-list'}>
        {items.map((item, i) => <li key={i}>{item}</li>)}
      </ul>
    </div>
  )
}

export default PaperCard
