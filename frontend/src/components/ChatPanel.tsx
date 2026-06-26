import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from '../api/client'

interface Source {
  id: string
  doc_id: string
  text: string
  score: number
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
}

export function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState<number | null>(null)
  const end = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const send = async () => {
    const text = input.trim()
    if (!text || loading) return
    setInput('')
    setMessages(m => [...m, { role: 'user', content: text }])
    setLoading(true)
    try {
      const data = await api.chat(text)
      setMessages(m => [...m, {
        role: 'assistant',
        content: data.answer || 'No response.',
        sources: data.sources,
      }])
    } catch {
      setMessages(m => [...m, { role: 'assistant', content: 'Error.' }])
    }
    setLoading(false)
    inputRef.current?.focus()
  }

  const clear = async () => { await api.clearChat(); setMessages([]) }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px 4px' }}>
        {messages.length === 0 && !loading && (
          <div className="muted" style={{ padding: 16, fontSize: 13, textAlign: 'center' }}>
            Ask about anything in this workspace.
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: 14 }}>
            <div className="label-muted" style={{ marginBottom: 4 }}>
              {m.role === 'user' ? 'You' : 'Loom'}
            </div>
            <div style={{
              fontSize: 13,
              lineHeight: 1.55,
              color: 'var(--text-primary)',
            }}>
              {m.role === 'user'
                ? <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
                : <div className="doc-body" style={{ fontSize: 13 }}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                  </div>}
            </div>
            {m.sources && m.sources.length > 0 && (
              <div style={{ marginTop: 6 }}>
                <button
                  onClick={() => setExpanded(expanded === i ? null : i)}
                  style={{
                    background: 'transparent', border: 'none',
                    color: 'var(--text-secondary)', fontSize: 11,
                    padding: 0, textDecoration: 'underline', textDecorationColor: '#ccc',
                  }}
                >
                  {expanded === i ? 'hide' : 'show'} {m.sources.length} sources
                </button>
                {expanded === i && (
                  <div style={{ marginTop: 6 }}>
                    {m.sources.slice(0, 6).map((s, j) => (
                      <div key={j} style={{
                        fontSize: 11,
                        padding: '6px 8px',
                        background: 'var(--bg-canvas)',
                        borderRadius: 5,
                        marginBottom: 4,
                        color: 'var(--text-secondary)',
                      }}>
                        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, marginBottom: 2 }}>
                          {s.doc_id}
                        </div>
                        <div style={{ lineHeight: 1.5 }}>{s.text}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        {loading && <div className="muted" style={{ fontSize: 12, padding: '4px 4px' }}>thinking…</div>}
        <div ref={end} />
      </div>

      <div style={{ borderTop: '1px solid var(--border-soft)', padding: '10px 4px 0' }}>
        <div style={{
          display: 'flex', gap: 6, alignItems: 'flex-end',
          background: 'var(--bg-canvas)', borderRadius: 8, padding: 6,
        }}>
          <textarea
            ref={inputRef}
            rows={1}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
            }}
            placeholder="Ask…"
            style={{
              flex: 1, background: 'transparent', border: 'none',
              resize: 'none', fontSize: 13, padding: '4px 6px',
              outline: 'none', maxHeight: 100, color: 'var(--text-primary)',
            }}
            onInput={e => {
              const t = e.currentTarget
              t.style.height = 'auto'
              t.style.height = Math.min(t.scrollHeight, 100) + 'px'
            }}
          />
          {messages.length > 0 && (
            <button
              onClick={clear}
              title="Clear"
              style={{
                background: 'transparent', border: 'none',
                color: 'var(--text-muted)', fontSize: 14, padding: '4px 6px',
              }}
            >
              ×
            </button>
          )}
          <button
            onClick={send}
            disabled={!input.trim() || loading}
            style={{
              background: input.trim() && !loading ? 'var(--text-primary)' : 'var(--bg-pill)',
              color: input.trim() && !loading ? 'white' : 'var(--text-muted)',
              border: 'none',
              padding: '4px 10px',
              borderRadius: 6,
              fontSize: 12,
              fontWeight: 500,
            }}
          >
            send
          </button>
        </div>
      </div>
    </div>
  )
}
