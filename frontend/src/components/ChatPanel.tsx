import { useState, useRef, useEffect } from 'react'
import { api } from '../api/client'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Send, Loader2, Trash2, User, Bot } from 'lucide-react'

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
  const [expandedSources, setExpandedSources] = useState<number | null>(null)
  const messagesEnd = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: text }])
    setLoading(true)

    try {
      const data = await api.chat(text)
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer || 'No response generated.',
          sources: data.sources,
        },
      ])
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Error: Failed to get response.' },
      ])
    }
    setLoading(false)
    inputRef.current?.focus()
  }

  const clearChat = async () => {
    await api.clearChat()
    setMessages([])
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {messages.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center h-full text-text-muted">
            <Bot size={32} className="mb-2 opacity-20" />
            <p className="text-sm font-medium mb-0.5">Ask about your papers</p>
            <p className="text-xs text-text-muted text-center px-4">
              Query your knowledge graph for insights and connections.
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex gap-2.5 ${msg.role === 'user' ? 'justify-end' : ''}`}>
            {msg.role === 'assistant' && (
              <div className="w-6 h-6 rounded-full bg-accent/10 flex items-center justify-center shrink-0 mt-0.5">
                <Bot size={12} className="text-accent" />
              </div>
            )}
            <div
              className={`max-w-[90%] ${
                msg.role === 'user'
                  ? 'bg-accent text-white rounded-2xl rounded-tr-md px-3 py-2'
                  : 'bg-surface-1 border border-surface-3 rounded-2xl rounded-tl-md px-3 py-2.5'
              }`}
            >
              {msg.role === 'user' ? (
                <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
              ) : (
                <div className="prose text-sm text-text-primary">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                </div>
              )}

              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-2 pt-1.5 border-t border-surface-3">
                  <button
                    onClick={() => setExpandedSources(expandedSources === i ? null : i)}
                    className="text-xs text-accent hover:text-accent-dim"
                  >
                    {expandedSources === i ? 'Hide' : 'Show'} {msg.sources.length} sources
                  </button>
                  {expandedSources === i && (
                    <div className="mt-1.5 space-y-1">
                      {msg.sources.slice(0, 5).map((src, j) => (
                        <div key={j} className="text-xs bg-surface-2 rounded p-1.5">
                          <span className="text-accent font-mono text-[10px]">{src.doc_id}</span>
                          <p className="text-text-secondary mt-0.5 line-clamp-2">{src.text}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
            {msg.role === 'user' && (
              <div className="w-6 h-6 rounded-full bg-surface-2 flex items-center justify-center shrink-0 mt-0.5">
                <User size={12} className="text-text-secondary" />
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex gap-2.5">
            <div className="w-6 h-6 rounded-full bg-accent/10 flex items-center justify-center shrink-0">
              <Bot size={12} className="text-accent" />
            </div>
            <div className="bg-surface-1 border border-surface-3 rounded-2xl rounded-tl-md px-3 py-2.5">
              <Loader2 size={14} className="animate-spin text-accent" />
            </div>
          </div>
        )}

        <div ref={messagesEnd} />
      </div>

      {/* Input */}
      <div className="px-4 pb-3 pt-2 border-t border-surface-3">
        <div className="flex items-end gap-2 bg-surface-1 border border-surface-3 rounded-xl p-2">
          <textarea
            ref={inputRef}
            rows={1}
            className="flex-1 bg-transparent text-sm text-text-primary placeholder-text-muted resize-none focus:outline-none px-2 py-1 max-h-28"
            placeholder="Ask about your research..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onInput={e => {
              const t = e.currentTarget
              t.style.height = 'auto'
              t.style.height = Math.min(t.scrollHeight, 112) + 'px'
            }}
          />
          <div className="flex items-center gap-1 shrink-0">
            {messages.length > 0 && (
              <button
                onClick={clearChat}
                className="p-1.5 text-text-muted hover:text-red-500 rounded-md transition-colors"
                title="Clear chat"
              >
                <Trash2 size={13} />
              </button>
            )}
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className="p-1.5 bg-accent text-white rounded-lg hover:bg-accent-dim disabled:opacity-30 transition-colors"
            >
              <Send size={13} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
