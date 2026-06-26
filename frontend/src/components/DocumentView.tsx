import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api, DocumentDescriptor } from '../api/client'

interface DocumentViewProps {
  docId: string
}

interface BodyResponse {
  doc_id: string
  title: string
  doc_type: string
  source_url: string
  body: string
  metadata_status: string
}

export function DocumentView({ docId }: DocumentViewProps) {
  const [descriptor, setDescriptor] = useState<DocumentDescriptor | null>(null)
  const [body, setBody] = useState<BodyResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string>('')

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError('')
    setDescriptor(null)
    setBody(null)
    ;(async () => {
      try {
        const [d, b] = await Promise.all([
          api.document(docId),
          api.documentBody(docId),
        ])
        if (cancelled) return
        if (d?.detail) { setError(d.detail); return }
        setDescriptor(d as DocumentDescriptor)
        setBody(b as BodyResponse)
      } catch (e) {
        if (!cancelled) setError('failed to load document')
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => { cancelled = true }
  }, [docId])

  if (loading) {
    return <Centered><div className="muted" style={{ fontSize: 13 }}>loading…</div></Centered>
  }
  if (error) {
    return <Centered><div className="muted" style={{ fontSize: 13 }}>{error}</div></Centered>
  }
  if (!descriptor || !body) {
    return <Centered><div className="muted" style={{ fontSize: 13 }}>not found</div></Centered>
  }

  return (
    <article style={{
      maxWidth: 760,
      margin: '0 auto',
      padding: '48px 56px 80px',
    }}>
      <Header descriptor={descriptor} sourceUrl={body.source_url} />
      <div className="doc-body" style={{ marginTop: 28 }}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{body.body || ''}</ReactMarkdown>
      </div>
    </article>
  )
}

function Header({ descriptor, sourceUrl }: { descriptor: DocumentDescriptor; sourceUrl: string }) {
  const meta: string[] = []
  if (descriptor.authors && descriptor.authors.length) {
    const a = descriptor.authors.slice(0, 4).join(', ')
    meta.push(descriptor.authors.length > 4 ? `${a} et al.` : a)
  }
  if (descriptor.published_at) meta.push(descriptor.published_at)
  meta.push(descriptor.doc_type.replace(/_/g, ' '))

  return (
    <header>
      {sourceUrl && (
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 14 }}>
          Source:{' '}
          <a href={sourceUrl} target="_blank" rel="noreferrer">{sourceUrl}</a>
        </div>
      )}
      <h1 className="serif" style={{
        fontSize: 32,
        fontWeight: 600,
        lineHeight: 1.18,
        margin: 0,
        color: 'var(--text-primary)',
      }}>
        {descriptor.title || descriptor.doc_id}
      </h1>
      <div className="subtle" style={{ fontSize: 12, marginTop: 10 }}>
        {meta.join(' · ')}
      </div>
      {descriptor.metadata_status && descriptor.metadata_status !== 'derived' && (
        <div className="subtle" style={{ fontSize: 11, marginTop: 8, fontStyle: 'italic' }}>
          {descriptor.metadata_status === 'pending' || descriptor.metadata_status === 'deriving'
            ? 'deriving metadata…'
            : `metadata: ${descriptor.metadata_status}`}
        </div>
      )}
    </header>
  )
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      maxWidth: 760, margin: '0 auto', padding: '120px 56px',
      textAlign: 'center',
    }}>
      {children}
    </div>
  )
}
