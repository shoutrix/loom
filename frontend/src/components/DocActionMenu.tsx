import { useEffect, useRef, useState } from 'react'

interface DocActionMenuProps {
  docType: string
  onDelete: () => void
}

/**
 * Three-dot menu shown on hover in the contents tree.
 * Surface adapts to doc_type — only research_paper gets the
 * citation-tree / graph actions (and even those are placeholders for
 * the post-Phase B re-exposure).
 */
export function DocActionMenu({ docType, onDelete }: DocActionMenuProps) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  const isPaper = docType === 'research_paper'

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button
        onClick={e => { e.stopPropagation(); setOpen(o => !o) }}
        aria-label="More actions"
        style={{
          background: 'transparent',
          border: 'none',
          padding: '4px 6px',
          borderRadius: 4,
          color: 'var(--text-secondary)',
          fontSize: 16,
          lineHeight: 1,
        }}
      >
        ⋮
      </button>
      {open && (
        <div
          className="menu"
          style={{ position: 'absolute', right: 0, top: '100%', zIndex: 30 }}
        >
          {isPaper && (
            <>
              <button className="menu-item" disabled style={{ opacity: 0.4 }}>
                Citation tree
              </button>
              <button className="menu-item" disabled style={{ opacity: 0.4 }}>
                Explore graph
              </button>
            </>
          )}
          <button
            className="menu-item"
            data-danger="true"
            onClick={() => { setOpen(false); onDelete() }}
          >
            Delete
          </button>
        </div>
      )}
    </div>
  )
}
