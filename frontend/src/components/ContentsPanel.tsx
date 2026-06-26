import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { DocActionMenu } from './DocActionMenu'
import { WorkspaceSwitcher } from './WorkspaceSwitcher'

interface DocSummary {
  doc_id: string
  doc_type: string
  title: string
  tldr?: string
  published_at?: string
  metadata_status?: string
}

interface CategoryNode {
  category: string
  paper_count: number
  papers: DocSummary[]
  subcategories: CategoryNode[]
}

interface ContentsTree {
  total_papers: number
  uncategorized_count: number
  contents: CategoryNode[]
  uncategorized: DocSummary[]
}

interface ContentsPanelProps {
  workspaceId: string
  activeDocId: string | null
  onSelectDoc: (docId: string) => void
  onSelectNewNote: () => void
  onSelectBrief: () => void
  onWorkspaceSwitch: (id: string) => void
}

export function ContentsPanel({
  workspaceId,
  activeDocId,
  onSelectDoc,
  onSelectNewNote,
  onSelectBrief,
  onWorkspaceSwitch,
}: ContentsPanelProps) {
  const [tree, setTree] = useState<ContentsTree | null>(null)
  const [loading, setLoading] = useState(true)

  const load = async () => {
    try {
      const data = await api.documentsContents()
      setTree({
        total_papers: data.total_papers || 0,
        uncategorized_count: data.uncategorized_count || 0,
        contents: data.contents || [],
        uncategorized: data.uncategorized || [],
      })
    } catch {
      setTree({ total_papers: 0, uncategorized_count: 0, contents: [], uncategorized: [] })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [workspaceId])

  const onDelete = async (docId: string, title: string) => {
    if (!confirm(`Delete "${title}"?\nThis cannot be undone.`)) return
    try { await api.deleteDocument(docId); load() } catch {}
  }

  return (
    <aside className="panel" style={{
      width: 280,
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      <div style={{ padding: '10px 8px 6px' }}>
        <WorkspaceSwitcher
          currentId={workspaceId}
          onSwitch={onWorkspaceSwitch}
        />
      </div>

      <div style={{ padding: '4px 8px 8px', borderBottom: '1px solid var(--border-soft)' }}>
        <Row label="+  New note" onClick={onSelectNewNote} />
        <Row label="☰  Brief" onClick={onSelectBrief} />
      </div>

      <div style={{ padding: '14px 12px 4px' }}>
        <div className="label-muted">Contents</div>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '0 6px 12px' }}>
        {loading && <div className="muted" style={{ padding: 12, fontSize: 13 }}>loading…</div>}
        {!loading && tree && tree.total_papers === 0 && (
          <div className="muted" style={{ padding: 12, fontSize: 13 }}>
            No documents yet. Submit one to get started.
          </div>
        )}
        {!loading && tree && tree.contents.map(node => (
          <CategoryBranch
            key={node.category}
            node={node}
            depth={0}
            activeDocId={activeDocId}
            onSelectDoc={onSelectDoc}
            onDelete={onDelete}
          />
        ))}
        {!loading && tree && tree.uncategorized.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <CategoryHeader name="Uncategorized" depth={0} />
            {tree.uncategorized.map(d => (
              <DocRow
                key={d.doc_id}
                doc={d}
                depth={1}
                active={d.doc_id === activeDocId}
                onClick={() => onSelectDoc(d.doc_id)}
                onDelete={() => onDelete(d.doc_id, d.title)}
              />
            ))}
          </div>
        )}
      </div>
    </aside>
  )
}

function Row({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="row-hover"
      style={{
        width: '100%',
        background: 'transparent',
        border: 'none',
        textAlign: 'left',
        padding: '8px 10px',
        fontSize: 13,
        color: 'var(--text-primary)',
      }}
    >
      {label}
    </button>
  )
}

function CategoryHeader({ name, depth }: { name: string; depth: number }) {
  return (
    <div
      className="subtle"
      style={{
        fontSize: 12,
        fontWeight: 500,
        padding: '6px 8px 2px',
        paddingLeft: 8 + depth * 12,
        letterSpacing: 0.02,
      }}
    >
      {name}
    </div>
  )
}

function CategoryBranch({
  node,
  depth,
  activeDocId,
  onSelectDoc,
  onDelete,
}: {
  node: CategoryNode
  depth: number
  activeDocId: string | null
  onSelectDoc: (id: string) => void
  onDelete: (id: string, title: string) => void
}) {
  return (
    <div>
      <CategoryHeader name={node.category} depth={depth} />
      {node.papers.map(d => (
        <DocRow
          key={d.doc_id}
          doc={d}
          depth={depth + 1}
          active={d.doc_id === activeDocId}
          onClick={() => onSelectDoc(d.doc_id)}
          onDelete={() => onDelete(d.doc_id, d.title)}
        />
      ))}
      {node.subcategories.map(sub => (
        <CategoryBranch
          key={sub.category}
          node={sub}
          depth={depth + 1}
          activeDocId={activeDocId}
          onSelectDoc={onSelectDoc}
          onDelete={onDelete}
        />
      ))}
    </div>
  )
}

function DocRow({
  doc,
  depth,
  active,
  onClick,
  onDelete,
}: {
  doc: DocSummary
  depth: number
  active: boolean
  onClick: () => void
  onDelete: () => void
}) {
  const [hover, setHover] = useState(false)
  const indent = 8 + depth * 12
  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      className="row-hover"
      style={{
        display: 'flex',
        alignItems: 'center',
        padding: '6px 4px 6px 0',
        paddingLeft: indent,
        background: active ? 'var(--bg-panel-hover)' : 'transparent',
        position: 'relative',
      }}
    >
      <button
        onClick={onClick}
        style={{
          flex: 1,
          background: 'transparent',
          border: 'none',
          textAlign: 'left',
          padding: '0 6px',
          fontSize: 13,
          fontWeight: active ? 500 : 400,
          color: 'var(--text-primary)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
        title={doc.title}
      >
        {doc.title || doc.doc_id}
      </button>
      <div style={{ visibility: hover ? 'visible' : 'hidden' }}>
        <DocActionMenu docType={doc.doc_type} onDelete={onDelete} />
      </div>
    </div>
  )
}
