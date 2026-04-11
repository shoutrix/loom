import { useState, useEffect, useRef } from 'react'
import { api } from '../api/client'
import { ChevronDown, Plus, Check, Trash2 } from 'lucide-react'

interface Workspace {
  workspace_id: string
  active: boolean
  description: string
}

interface Props {
  currentId: string
  onSwitch: (id: string, displayName?: string) => void
}

export function WorkspaceSwitcher({ currentId, onSwitch }: Props) {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([])
  const [open, setOpen] = useState(false)
  const [creating, setCreating] = useState(false)
  const [newId, setNewId] = useState('')
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (open) load()
  }, [open])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const load = async () => {
    try {
      const data = await api.workspaces()
      setWorkspaces(data || [])
    } catch {
      // ignore
    }
  }

  const switchTo = async (id: string) => {
    if (id === currentId) { setOpen(false); return }
    try {
      await api.switchWorkspace(id)
      onSwitch(id)
      setOpen(false)
      window.location.reload()
    } catch {
      // ignore
    }
  }

  const create = async () => {
    if (!newId.trim()) return
    try {
      await api.createWorkspace(newId.trim())
      await switchTo(newId.trim())
    } catch {
      // ignore
    }
  }

  const deleteWs = async (id: string) => {
    if (id === currentId) return
    if (!confirm(`Delete workspace "${id}" and all its data? This cannot be undone.`)) return
    try {
      await api.deleteWorkspace(id)
      load()
    } catch {
      // ignore
    }
  }

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="p-1 text-text-muted hover:text-text-secondary transition-colors"
        title="Switch workspace"
      >
        <ChevronDown size={14} />
      </button>

      {open && (
        <div className="absolute left-0 top-full mt-1 w-52 bg-white border border-surface-3 rounded-lg shadow-xl z-50 overflow-hidden">
          <div className="py-1">
            {workspaces.map(ws => (
              <div
                key={ws.workspace_id}
                className="group flex items-center hover:bg-surface-2 transition-colors"
              >
                <button
                  onClick={() => switchTo(ws.workspace_id)}
                  className="flex-1 flex items-center gap-2 px-3 py-2 text-sm"
                >
                  {ws.workspace_id === currentId ? (
                    <Check size={13} className="text-accent" />
                  ) : (
                    <span className="w-[13px]" />
                  )}
                  <span className={ws.workspace_id === currentId ? 'text-accent font-medium' : 'text-text-secondary'}>
                    {ws.workspace_id}
                  </span>
                </button>
                {ws.workspace_id !== currentId && (
                  <button
                    onClick={() => deleteWs(ws.workspace_id)}
                    className="opacity-0 group-hover:opacity-100 px-2 py-1 text-text-muted hover:text-red-500 transition-all"
                    title="Delete workspace"
                  >
                    <Trash2 size={12} />
                  </button>
                )}
              </div>
            ))}
          </div>
          <div className="border-t border-surface-3 p-2">
            {creating ? (
              <div className="flex gap-1">
                <input
                  className="flex-1 bg-surface-1 border border-surface-3 rounded px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent"
                  placeholder="workspace-name"
                  value={newId}
                  onChange={e => setNewId(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && create()}
                  autoFocus
                />
                <button onClick={create} className="text-xs bg-accent text-white px-2 py-1 rounded">
                  Create
                </button>
              </div>
            ) : (
              <button
                onClick={() => setCreating(true)}
                className="w-full flex items-center gap-2 px-2 py-1.5 text-xs text-text-muted hover:text-accent transition-colors"
              >
                <Plus size={12} /> New workspace
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
