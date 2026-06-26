import { useState } from 'react'
import { ChatPanel } from './ChatPanel'
import { GraphView } from './GraphView'
import { PillToggle } from './PillToggle'

type Mode = 'chat' | 'graph'

export function RightRail() {
  const [mode, setMode] = useState<Mode>('chat')

  return (
    <aside className="panel" style={{
      width: 360,
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      <div style={{ padding: 10, borderBottom: '1px solid var(--border-soft)' }}>
        <PillToggle<Mode>
          options={[
            { value: 'chat', label: 'Chat' },
            { value: 'graph', label: 'Graph' },
          ]}
          value={mode}
          onChange={setMode}
        />
      </div>
      <div style={{ flex: 1, minHeight: 0, padding: '8px 10px 12px' }}>
        {mode === 'chat' ? <ChatPanel /> : <GraphView />}
      </div>
    </aside>
  )
}
