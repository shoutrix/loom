import { useEffect, useState } from 'react'
import { api } from './api/client'
import { BriefPanel } from './components/BriefPanel'
import { ContentsPanel } from './components/ContentsPanel'
import { DocumentView } from './components/DocumentView'
import { NoteEditor } from './components/NoteEditor'
import { RightRail } from './components/RightRail'

type CenterMode =
  | { kind: 'doc'; docId: string }
  | { kind: 'note-new' }
  | { kind: 'brief' }
  | { kind: 'empty' }

export default function App() {
  const [workspaceId, setWorkspaceId] = useState<string>('default')
  const [mode, setMode] = useState<CenterMode>({ kind: 'empty' })
  const [contentsKey, setContentsKey] = useState(0)

  useEffect(() => {
    api.activeWorkspace()
      .then(r => { if (r?.workspace_id) setWorkspaceId(r.workspace_id) })
      .catch(() => {})
  }, [])

  const selectDoc = (docId: string) => setMode({ kind: 'doc', docId })
  const openNewNote = () => setMode({ kind: 'note-new' })
  const openBrief = () => setMode({ kind: 'brief' })
  const refreshContents = () => setContentsKey(k => k + 1)

  return (
    <div style={{
      height: '100vh',
      width: '100vw',
      background: 'var(--bg-canvas)',
      display: 'grid',
      gridTemplateColumns: '24px 280px 1fr 360px 24px',
      gridTemplateRows: '24px 1fr 24px',
      gap: 16,
      overflow: 'hidden',
    }}>
      <div style={{ gridColumn: 2, gridRow: 2 }}>
        <ContentsPanel
          key={contentsKey}
          workspaceId={workspaceId}
          activeDocId={mode.kind === 'doc' ? mode.docId : null}
          onSelectDoc={selectDoc}
          onSelectNewNote={openNewNote}
          onSelectBrief={openBrief}
          onWorkspaceSwitch={setWorkspaceId}
        />
      </div>

      <main style={{
        gridColumn: 3, gridRow: 2,
        height: '100%',
        overflowY: 'auto',
        padding: '8px 0',
      }}>
        <Center
          mode={mode}
          onSubmittedNote={docId => {
            refreshContents()
            setMode({ kind: 'doc', docId })
          }}
          onCancelNote={() => setMode({ kind: 'empty' })}
        />
      </main>

      <div style={{ gridColumn: 4, gridRow: 2 }}>
        <RightRail />
      </div>
    </div>
  )
}

function Center({
  mode,
  onSubmittedNote,
  onCancelNote,
}: {
  mode: CenterMode
  onSubmittedNote: (docId: string) => void
  onCancelNote: () => void
}) {
  if (mode.kind === 'doc') return <DocumentView docId={mode.docId} />
  if (mode.kind === 'note-new')
    return <NoteEditor onSubmitted={onSubmittedNote} onCancel={onCancelNote} />
  if (mode.kind === 'brief') return <BriefPanel />
  return (
    <div style={{
      maxWidth: 760, margin: '0 auto', padding: '160px 56px',
      textAlign: 'center', color: 'var(--text-muted)', fontSize: 14,
    }}>
      Select a document, write a new note, or open the workspace brief.
    </div>
  )
}
