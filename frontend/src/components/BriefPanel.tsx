import { useEffect, useState } from 'react'
import { api } from '../api/client'

interface Brief {
  goal: string
  scope: string
  key_questions: string[]
  current_focus: string
  exclude: string
}

interface BriefPayload {
  exists: boolean
  workspace_id: string
  brief?: Brief
  user_notes?: string
  generated_at?: string
  generated_from_paper_count?: number
}

const EMPTY: Brief = {
  goal: '', scope: '', key_questions: [], current_focus: '', exclude: '',
}

/**
 * Workspace brief — opens in the center column. View-mode by default,
 * with an inline "edit" toggle that drops into editable textareas.
 */
export function BriefPanel() {
  const [payload, setPayload] = useState<BriefPayload | null>(null)
  const [draft, setDraft] = useState<Brief>(EMPTY)
  const [notes, setNotes] = useState('')
  const [kqText, setKqText] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [editing, setEditing] = useState(false)
  const [regenerating, setRegenerating] = useState(false)

  const hydrate = (p: BriefPayload) => {
    const b = p.brief ?? EMPTY
    setDraft(b)
    setNotes(p.user_notes ?? '')
    setKqText((b.key_questions ?? []).join('\n'))
  }

  const load = async () => {
    setLoading(true)
    try {
      const p = await api.workspaceBrief()
      setPayload(p); hydrate(p)
    } catch {}
    setLoading(false)
  }

  useEffect(() => { load() }, [])

  const save = async () => {
    setSaving(true)
    try {
      const kq = kqText.split('\n').map(s => s.trim()).filter(Boolean)
      await api.setWorkspaceBrief({
        brief: { ...draft, key_questions: kq },
        user_notes: notes,
      })
      await load()
      setEditing(false)
    } catch {}
    setSaving(false)
  }

  const regenerate = async () => {
    setRegenerating(true)
    try { await api.regenerateWorkspaceBrief(); await load() } catch {}
    setRegenerating(false)
  }

  if (loading) {
    return <Wrapper><div className="muted" style={{ fontSize: 13 }}>loading…</div></Wrapper>
  }

  return (
    <Wrapper>
      <header style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 18 }}>
        <h1 className="serif" style={{ fontSize: 28, fontWeight: 600, margin: 0 }}>Brief</h1>
        <div style={{ display: 'flex', gap: 10 }}>
          {!editing && (
            <button onClick={() => setEditing(true)} className="muted" style={btnLink}>edit</button>
          )}
          {editing && (
            <>
              <button onClick={() => { hydrate(payload!); setEditing(false) }} className="muted" style={btnLink}>cancel</button>
              <button onClick={save} disabled={saving} style={btnPrimary}>{saving ? 'saving…' : 'save'}</button>
            </>
          )}
          <button onClick={regenerate} disabled={regenerating} className="muted" style={btnLink}>
            {regenerating ? 'regenerating…' : 'regenerate'}
          </button>
        </div>
      </header>

      {!editing ? (
        <ReadOnly brief={draft} notes={notes} />
      ) : (
        <Form
          brief={draft}
          setBrief={setDraft}
          notes={notes}
          setNotes={setNotes}
          kqText={kqText}
          setKqText={setKqText}
        />
      )}

      {payload?.generated_at && (
        <div className="subtle" style={{ marginTop: 32, fontSize: 11 }}>
          Last regenerated {payload.generated_at}
          {payload.generated_from_paper_count != null
            ? ` · ${payload.generated_from_paper_count} ingested`
            : ''}
        </div>
      )}
    </Wrapper>
  )
}

function ReadOnly({ brief, notes }: { brief: Brief; notes: string }) {
  return (
    <div className="doc-body">
      <Field label="Goal" value={brief.goal} />
      <Field label="Scope" value={brief.scope} />
      <Field label="Current focus" value={brief.current_focus} />
      <Field label="Exclude" value={brief.exclude} />
      {brief.key_questions.length > 0 && (
        <>
          <div className="label-muted" style={{ marginTop: 18, marginBottom: 8 }}>Key questions</div>
          <ul>
            {brief.key_questions.map((q, i) => <li key={i}>{q}</li>)}
          </ul>
        </>
      )}
      {notes && (
        <>
          <div className="label-muted" style={{ marginTop: 24, marginBottom: 8 }}>Notes</div>
          <div style={{ whiteSpace: 'pre-wrap' }}>{notes}</div>
        </>
      )}
    </div>
  )
}

function Field({ label, value }: { label: string; value: string }) {
  if (!value) return null
  return (
    <div style={{ marginBottom: 18 }}>
      <div className="label-muted" style={{ marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 15, lineHeight: 1.6 }}>{value}</div>
    </div>
  )
}

function Form({
  brief, setBrief, notes, setNotes, kqText, setKqText,
}: {
  brief: Brief
  setBrief: (b: Brief) => void
  notes: string
  setNotes: (s: string) => void
  kqText: string
  setKqText: (s: string) => void
}) {
  const text = (label: string, val: string, set: (v: string) => void, rows = 2) => (
    <div style={{ marginBottom: 16 }}>
      <div className="label-muted" style={{ marginBottom: 6 }}>{label}</div>
      <textarea
        value={val}
        onChange={e => set(e.target.value)}
        rows={rows}
        style={editorStyle}
      />
    </div>
  )
  return (
    <div>
      {text('Goal', brief.goal, v => setBrief({ ...brief, goal: v }))}
      {text('Scope', brief.scope, v => setBrief({ ...brief, scope: v }))}
      {text('Current focus', brief.current_focus, v => setBrief({ ...brief, current_focus: v }))}
      {text('Exclude', brief.exclude, v => setBrief({ ...brief, exclude: v }))}
      {text('Key questions (one per line)', kqText, setKqText, 4)}
      {text('Notes', notes, setNotes, 6)}
    </div>
  )
}

const editorStyle: React.CSSProperties = {
  width: '100%',
  fontFamily: 'var(--font-sans)',
  fontSize: 14,
  lineHeight: 1.6,
  padding: '10px 12px',
  border: '1px solid #d9d9d9',
  borderRadius: 8,
  background: 'transparent',
  color: 'var(--text-primary)',
  resize: 'vertical',
  outline: 'none',
}

const btnLink: React.CSSProperties = {
  background: 'transparent', border: 'none', fontSize: 12,
  textDecoration: 'underline', textDecorationColor: '#ccc',
  color: 'var(--text-secondary)',
}

const btnPrimary: React.CSSProperties = {
  background: 'var(--text-primary)', color: 'white', border: 'none',
  borderRadius: 6, padding: '5px 12px', fontSize: 12, fontWeight: 500,
}

function Wrapper({ children }: { children: React.ReactNode }) {
  return (
    <article style={{
      maxWidth: 760, margin: '0 auto', padding: '48px 56px 64px',
    }}>
      {children}
    </article>
  )
}
