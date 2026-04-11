import { useState, useEffect, useCallback, useRef } from 'react'
import { SourcesPanel } from './components/SourcesPanel'
import { AddSourcesModal } from './components/AddSourcesModal'
import { ChatPanel } from './components/ChatPanel'
import { GraphView } from './components/GraphView'
import { NoteEditor } from './components/NoteEditor'
import { PaperViewer } from './components/PaperViewer'
import { QueueStatus } from './components/QueueStatus'
import { WorkspaceSwitcher } from './components/WorkspaceSwitcher'
import { api } from './api/client'
import {
  Network, MessageCircle, X, FileText,
} from 'lucide-react'

interface PaperMeta {
  paper_id: string
  title: string
  arxiv_id?: string
  doi?: string
}

interface Tab {
  id: string
  label: string
  type: 'notes' | 'paper'
  paperMeta?: PaperMeta
}

type RightView = 'graph' | 'chat'

function usePanelResize(
  initialWidth: number,
  minWidth: number,
  maxWidth: number,
  side: 'left' | 'right',
) {
  const [width, setWidth] = useState(initialWidth)
  const dragging = useRef(false)
  const startX = useRef(0)
  const startW = useRef(0)

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    dragging.current = true
    startX.current = e.clientX
    startW.current = width

    const onMouseMove = (ev: MouseEvent) => {
      if (!dragging.current) return
      const delta = ev.clientX - startX.current
      const newW = side === 'left'
        ? startW.current + delta
        : startW.current - delta
      setWidth(Math.max(minWidth, Math.min(maxWidth, newW)))
    }

    const onMouseUp = () => {
      dragging.current = false
      document.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('mouseup', onMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }

    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('mouseup', onMouseUp)
  }, [width, minWidth, maxWidth, side])

  return { width, onMouseDown }
}

export default function App() {
  const [tabs, setTabs] = useState<Tab[]>([
    { id: 'notes', label: 'Notes', type: 'notes' },
  ])
  const [activeTab, setActiveTab] = useState('notes')
  const [rightView, setRightView] = useState<RightView>('chat')
  const [showAddSources, setShowAddSources] = useState(false)
  const [refreshKey, setRefreshKey] = useState(0)

  const [workspaceId, setWorkspaceId] = useState('')
  const [workspaceName, setWorkspaceName] = useState('')
  const [editingName, setEditingName] = useState(false)
  const [nameInput, setNameInput] = useState('')

  const left = usePanelResize(280, 200, 480, 'left')
  const right = usePanelResize(380, 260, 600, 'right')

  useEffect(() => {
    api.activeWorkspace().then((data: any) => {
      setWorkspaceId(data.workspace_id || 'default')
      setWorkspaceName(data.display_name || data.workspace_id || 'default')
    })
  }, [])

  const refresh = () => setRefreshKey(k => k + 1)

  const openPaperTab = useCallback((meta: PaperMeta) => {
    setTabs(prev => {
      const existing = prev.find(t => t.id === meta.paper_id)
      if (existing) return prev
      return [...prev, {
        id: meta.paper_id,
        label: meta.title.length > 30 ? meta.title.slice(0, 30) + '...' : meta.title,
        type: 'paper' as const,
        paperMeta: meta,
      }]
    })
    setActiveTab(meta.paper_id)
  }, [])

  const closeTab = useCallback((tabId: string) => {
    if (tabId === 'notes') return
    setTabs(prev => prev.filter(t => t.id !== tabId))
    setActiveTab(prev => prev === tabId ? 'notes' : prev)
  }, [])

  const saveName = async () => {
    const trimmed = nameInput.trim()
    if (trimmed && trimmed !== workspaceName) {
      await api.renameWorkspace(trimmed)
      setWorkspaceName(trimmed)
    }
    setEditingName(false)
  }

  const currentTab = tabs.find(t => t.id === activeTab)

  return (
    <div className="h-screen flex flex-col bg-surface-0 text-text-primary">
      {/* Header */}
      <header className="h-12 flex items-center justify-between px-5 border-b border-surface-3 bg-white shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold tracking-tight text-text-primary">Loom</span>
          <span className="text-surface-3 select-none">/</span>
          {editingName ? (
            <input
              className="text-sm font-medium bg-surface-1 border border-surface-3 rounded px-2 py-0.5 text-text-primary focus:outline-none focus:border-accent w-48"
              value={nameInput}
              onChange={e => setNameInput(e.target.value)}
              onBlur={saveName}
              onKeyDown={e => { if (e.key === 'Enter') saveName(); if (e.key === 'Escape') setEditingName(false) }}
              autoFocus
            />
          ) : (
            <button
              onClick={() => { setNameInput(workspaceName); setEditingName(true) }}
              className="text-sm font-medium text-text-secondary hover:text-text-primary transition-colors"
              title="Click to rename workspace"
            >
              {workspaceName}
            </button>
          )}
          <WorkspaceSwitcher
            currentId={workspaceId}
            onSwitch={(id, name) => { setWorkspaceId(id); setWorkspaceName(name || id) }}
          />
        </div>
        <QueueStatus />
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Left panel: Sources */}
        <aside
          className="flex flex-col border-r border-surface-3 bg-surface-1 shrink-0"
          style={{ width: left.width }}
        >
          <SourcesPanel
            onOpenPaper={openPaperTab}
            onAddSources={() => setShowAddSources(true)}
            key={refreshKey}
          />
        </aside>

        {/* Left resize handle */}
        <div
          className="w-1 cursor-col-resize hover:bg-accent/30 active:bg-accent/50 transition-colors shrink-0"
          onMouseDown={left.onMouseDown}
        />

        {/* Center panel: Tabs */}
        <main className="flex-1 flex flex-col overflow-hidden min-w-0">
          {/* Tab bar */}
          <div className="flex items-end gap-0 px-2 pt-1 bg-surface-1 border-b border-surface-3 shrink-0 overflow-x-auto">
            {tabs.map(tab => (
              <div
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`group flex items-center gap-1.5 px-3 py-1.5 text-sm cursor-pointer rounded-t-lg border border-b-0 transition-colors max-w-[200px] ${
                  activeTab === tab.id
                    ? 'bg-white border-surface-3 text-text-primary font-medium -mb-px z-10'
                    : 'bg-surface-2 border-transparent text-text-secondary hover:text-text-primary hover:bg-surface-1'
                }`}
              >
                {tab.type === 'notes' ? (
                  <FileText size={13} className="shrink-0" />
                ) : (
                  <span className="w-2 h-2 rounded-full bg-accent shrink-0" />
                )}
                <span className="truncate text-xs">{tab.label}</span>
                {tab.id !== 'notes' && (
                  <button
                    onClick={e => { e.stopPropagation(); closeTab(tab.id) }}
                    className="opacity-0 group-hover:opacity-100 text-text-muted hover:text-text-primary ml-1 shrink-0 transition-opacity"
                  >
                    <X size={12} />
                  </button>
                )}
              </div>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-hidden bg-white">
            {currentTab?.type === 'notes' && <NoteEditor />}
            {currentTab?.type === 'paper' && currentTab.paperMeta && (
              <PaperViewer paperId={currentTab.paperMeta.paper_id} title={currentTab.paperMeta.title} />
            )}
          </div>
        </main>

        {/* Right resize handle */}
        <div
          className="w-1 cursor-col-resize hover:bg-accent/30 active:bg-accent/50 transition-colors shrink-0"
          onMouseDown={right.onMouseDown}
        />

        {/* Right panel: Graph / Chat toggle */}
        <aside
          className="flex flex-col border-l border-surface-3 bg-surface-1 shrink-0"
          style={{ width: right.width }}
        >
          <div className="flex border-b border-surface-3 shrink-0">
            <button
              onClick={() => setRightView('graph')}
              className={`flex-1 py-2.5 text-sm font-medium flex items-center justify-center gap-1.5 transition-colors ${
                rightView === 'graph'
                  ? 'text-accent border-b-2 border-accent'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              <Network size={14} /> Graph
            </button>
            <button
              onClick={() => setRightView('chat')}
              className={`flex-1 py-2.5 text-sm font-medium flex items-center justify-center gap-1.5 transition-colors ${
                rightView === 'chat'
                  ? 'text-accent border-b-2 border-accent'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              <MessageCircle size={14} /> Chat
            </button>
          </div>
          <div className="flex-1 overflow-hidden">
            {rightView === 'graph' ? (
              <GraphView key={refreshKey} />
            ) : (
              <ChatPanel />
            )}
          </div>
        </aside>
      </div>

      {/* Add Sources modal */}
      {showAddSources && (
        <AddSourcesModal
          onClose={() => { setShowAddSources(false); refresh() }}
        />
      )}
    </div>
  )
}
