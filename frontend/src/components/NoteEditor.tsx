import { useState, useEffect } from 'react'
import { api } from '../api/client'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { FileText, Plus, Save, Eye, Edit3, Loader2 } from 'lucide-react'

interface VaultFile {
  path: string
  name: string
  size: number
}

export function NoteEditor() {
  const [files, setFiles] = useState<VaultFile[]>([])
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [content, setContent] = useState('')
  const [originalContent, setOriginalContent] = useState('')
  const [mode, setMode] = useState<'edit' | 'preview'>('preview')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [newTitle, setNewTitle] = useState('')
  const [showNew, setShowNew] = useState(false)

  useEffect(() => { loadFiles() }, [])

  const loadFiles = async () => {
    try {
      const data = await api.vaultFiles()
      setFiles(data.files || [])
    } catch {
      // ignore
    }
  }

  const loadFile = async (path: string) => {
    setLoading(true)
    try {
      const data = await api.vaultRead(path)
      setContent(data.content || '')
      setOriginalContent(data.content || '')
      setSelectedFile(path)
      setMode('preview')
    } catch {
      // ignore
    }
    setLoading(false)
  }

  const saveFile = async () => {
    if (!selectedFile) return
    setSaving(true)
    try {
      await api.vaultWrite(selectedFile, content)
      setOriginalContent(content)
    } catch {
      // ignore
    }
    setSaving(false)
  }

  const createNote = async () => {
    if (!newTitle.trim()) return
    try {
      await api.vaultNote(newTitle, `# ${newTitle}\n\n`)
      setNewTitle('')
      setShowNew(false)
      await loadFiles()
    } catch {
      // ignore
    }
  }

  const hasChanges = content !== originalContent

  return (
    <div className="flex h-full">
      {/* File sidebar */}
      <div className="w-52 border-r border-surface-3 flex flex-col shrink-0 bg-surface-1">
        <div className="flex items-center justify-between px-3 py-2 border-b border-surface-3">
          <span className="text-xs font-medium text-text-secondary uppercase tracking-wide">Notes</span>
          <button onClick={() => setShowNew(!showNew)} className="text-text-muted hover:text-accent transition-colors">
            <Plus size={14} />
          </button>
        </div>

        {showNew && (
          <div className="p-2 border-b border-surface-3">
            <input
              className="w-full bg-white border border-surface-3 rounded px-2 py-1 text-xs text-text-primary placeholder-text-muted focus:outline-none focus:border-accent"
              placeholder="Note title..."
              value={newTitle}
              onChange={e => setNewTitle(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && createNote()}
              autoFocus
            />
          </div>
        )}

        <div className="flex-1 overflow-y-auto">
          {files.map(f => (
            <button
              key={f.path}
              onClick={() => loadFile(f.path)}
              className={`w-full text-left px-3 py-2 text-xs flex items-center gap-2 hover:bg-surface-2 transition-colors border-b border-surface-3/50 ${
                selectedFile === f.path ? 'bg-surface-2 text-text-primary font-medium' : 'text-text-secondary'
              }`}
            >
              <FileText size={12} className="shrink-0" />
              <span className="truncate">{f.name}</span>
            </button>
          ))}
          {files.length === 0 && (
            <p className="text-xs text-text-muted px-3 py-4 text-center">No notes yet</p>
          )}
        </div>
      </div>

      {/* Editor/preview */}
      <div className="flex-1 flex flex-col bg-white">
        {selectedFile ? (
          <>
            <div className="flex items-center justify-between px-4 py-2 border-b border-surface-3 shrink-0">
              <span className="text-sm text-text-secondary truncate">{selectedFile}</span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setMode(mode === 'edit' ? 'preview' : 'edit')}
                  className={`p-1.5 rounded-md transition-colors ${
                    mode === 'edit' ? 'bg-surface-2 text-accent' : 'text-text-muted hover:text-text-secondary'
                  }`}
                >
                  {mode === 'edit' ? <Eye size={14} /> : <Edit3 size={14} />}
                </button>
                {hasChanges && (
                  <button
                    onClick={saveFile}
                    disabled={saving}
                    className="flex items-center gap-1 px-2 py-1 bg-accent text-white text-xs rounded-md hover:bg-accent-dim disabled:opacity-50"
                  >
                    {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
                    Save
                  </button>
                )}
              </div>
            </div>
            <div className="flex-1 overflow-y-auto">
              {loading ? (
                <div className="flex items-center justify-center h-full">
                  <Loader2 size={20} className="animate-spin text-text-muted" />
                </div>
              ) : mode === 'edit' ? (
                <textarea
                  className="w-full h-full bg-transparent text-sm text-text-primary font-mono p-4 resize-none focus:outline-none"
                  value={content}
                  onChange={e => setContent(e.target.value)}
                />
              ) : (
                <div className="prose text-sm p-6 max-w-3xl mx-auto">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-text-muted">
            <div className="text-center">
              <FileText size={32} className="mx-auto mb-2 opacity-20" />
              <p className="text-sm">Select a note or create a new one</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
