import { useState } from 'react'
import { api } from '../api/client'

interface NoteEditorProps {
  onSubmitted: (docId: string) => void
  onCancel: () => void
}

/**
 * Inline note creator — opens in the center column.
 * Submits via `submit_document` with doc_type="note", so the new note
 * goes through the same metadata + ingestion pipeline as any other
 * content. No backdoor through the vault.
 */
export function NoteEditor({ onSubmitted, onCancel }: NoteEditorProps) {
  const [title, setTitle] = useState('')
  const [body, setBody] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const save = async () => {
    if (!body.trim()) { setError('body is required'); return }
    setSaving(true)
    setError('')
    try {
      const composed = title.trim() ? `# ${title.trim()}\n\n${body}` : body
      const res = await api.submitDocument({
        body: composed,
        doc_type: 'note',
        title: title.trim() || undefined,
      })
      if (res.ok) onSubmitted(res.doc_id)
      else setError(res.detail || res.error || 'failed to save')
    } catch {
      setError('failed to save')
    }
    setSaving(false)
  }

  return (
    <article style={{
      maxWidth: 760, margin: '0 auto', padding: '40px 56px 56px',
    }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
        marginBottom: 18,
      }}>
        <h1 className="serif" style={{ fontSize: 26, fontWeight: 600, margin: 0 }}>New note</h1>
        <div style={{ display: 'flex', gap: 12 }}>
          <button onClick={onCancel} style={btnLink}>cancel</button>
          <button
            onClick={save}
            disabled={saving || !body.trim()}
            style={{ ...btnPrimary, opacity: !body.trim() ? 0.4 : 1 }}
          >
            {saving ? 'saving…' : 'save'}
          </button>
        </div>
      </div>

      <input
        value={title}
        onChange={e => setTitle(e.target.value)}
        placeholder="Title (optional — derived from first heading if blank)"
        style={{
          width: '100%',
          fontFamily: 'var(--font-serif)',
          fontSize: 24,
          fontWeight: 600,
          border: 'none',
          outline: 'none',
          padding: '6px 0 12px',
          background: 'transparent',
          color: 'var(--text-primary)',
        }}
      />

      <textarea
        value={body}
        onChange={e => setBody(e.target.value)}
        placeholder="Write in markdown…"
        style={{
          width: '100%',
          fontFamily: 'var(--font-sans)',
          fontSize: 15,
          lineHeight: 1.65,
          minHeight: 480,
          border: 'none',
          outline: 'none',
          padding: '10px 0',
          background: 'transparent',
          color: 'var(--text-primary)',
          resize: 'vertical',
        }}
      />

      {error && (
        <div style={{ marginTop: 12, fontSize: 12, color: '#b00020' }}>{error}</div>
      )}
    </article>
  )
}

const btnLink: React.CSSProperties = {
  background: 'transparent', border: 'none', fontSize: 13,
  color: 'var(--text-secondary)',
  textDecoration: 'underline', textDecorationColor: '#ccc',
}

const btnPrimary: React.CSSProperties = {
  background: 'var(--text-primary)', color: 'white', border: 'none',
  borderRadius: 6, padding: '6px 14px', fontSize: 13, fontWeight: 500,
}
