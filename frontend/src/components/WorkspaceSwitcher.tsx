import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'

interface Workspace {
  workspace_id: string
  active: boolean
  description: string
}

interface Props {
  currentId: string
  onSwitch: (id: string) => void
}

/**
 * Grayscale workspace switcher anchored at the top of the left overlay.
 * Renders as the workspace name followed by a chevron that opens a
 * dropdown of every workspace + a "new workspace" row at the bottom.
 */
export function WorkspaceSwitcher({ currentId, onSwitch }: Props) {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([])
  const [open, setOpen] = useState(false)
  const [creating, setCreating] = useState(false)
  const [newId, setNewId] = useState('')
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => { if (open) load() }, [open])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const load = async () => {
    try { setWorkspaces((await api.workspaces()) || []) } catch {}
  }

  const switchTo = async (id: string) => {
    if (id === currentId) { setOpen(false); return }
    try {
      await api.switchWorkspace(id)
      onSwitch(id)
      setOpen(false)
      window.location.reload()
    } catch {}
  }

  const create = async () => {
    const id = newId.trim()
    if (!id) return
    try {
      await api.createWorkspace(id)
      await switchTo(id)
    } catch {}
  }

  const remove = async (id: string) => {
    if (id === currentId) return
    if (!confirm(`Delete workspace "${id}" and all its data?`)) return
    try { await api.deleteWorkspace(id); load() } catch {}
  }

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          width: '100%',
          background: 'transparent',
          border: 'none',
          padding: '8px 10px',
          borderRadius: 6,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          color: 'var(--text-primary)',
          fontSize: 14,
          fontWeight: 600,
        }}
        className="row-hover"
      >
        <span>{currentId}</span>
        <span className="subtle" style={{ fontSize: 11 }}>▾</span>
      </button>

      {open && (
        <div
          className="menu"
          style={{
            position: 'absolute', left: 0, top: '100%', marginTop: 4,
            width: '100%', zIndex: 40,
          }}
        >
          {workspaces.map(ws => (
            <div
              key={ws.workspace_id}
              style={{ display: 'flex', alignItems: 'center' }}
            >
              <button
                onClick={() => switchTo(ws.workspace_id)}
                className="menu-item"
                style={{ flex: 1, fontWeight: ws.workspace_id === currentId ? 600 : 400 }}
              >
                {ws.workspace_id === currentId ? '✓  ' : '    '}{ws.workspace_id}
              </button>
              {ws.workspace_id !== currentId && (
                <button
                  onClick={() => remove(ws.workspace_id)}
                  className="menu-item"
                  data-danger="true"
                  style={{ width: 32, padding: '7px 4px' }}
                  title="Delete workspace"
                >
                  ×
                </button>
              )}
            </div>
          ))}
          <div style={{ borderTop: '1px solid var(--border-soft)', marginTop: 4, paddingTop: 4 }}>
            {creating ? (
              <div style={{ display: 'flex', gap: 4, padding: '4px 6px' }}>
                <input
                  autoFocus
                  value={newId}
                  onChange={e => setNewId(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && create()}
                  placeholder="workspace-id"
                  style={{
                    flex: 1, padding: '5px 8px', fontSize: 13,
                    border: '1px solid var(--border-soft)', borderRadius: 5,
                    background: 'var(--bg-canvas)',
                  }}
                />
                <button onClick={create} className="menu-item" style={{ width: 'auto', padding: '5px 10px', fontWeight: 600 }}>
                  add
                </button>
              </div>
            ) : (
              <button onClick={() => setCreating(true)} className="menu-item">
                + New workspace
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
