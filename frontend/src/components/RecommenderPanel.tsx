import { useEffect, useState } from 'react'
import { api } from '../api/client'

type FeedItem = {
  id: string
  kind: 'paper' | 'blog'
  source: string
  url: string
  title: string
  authors: string[]
  published_at: string
  abstract: string
  features: Record<string, number>
  final_score: number
  calibrated_prob: number
  status: string
  exploration: boolean
  fetched_at: string
  run_id: string
}

type RecommenderProfile = {
  workspace_id: string
  description: string
  seed_topics: string[]
  pos_count: number
  neg_count: number
  ranker_stage: number
  config: Record<string, unknown>
  last_run_at: string
}

export default function RecommenderPanel({ workspaceId }: { workspaceId: string }) {
  const [profile, setProfile] = useState<RecommenderProfile | null>(null)
  const [items, setItems] = useState<FeedItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [materializing, setMaterializing] = useState<string | null>(null)

  async function refresh() {
    setLoading(true)
    setError(null)
    try {
      const [p, its] = await Promise.all([
        api.recommenderProfile(workspaceId),
        api.recommenderItems(workspaceId, { status: statusFilter || undefined, limit: 100 }),
      ])
      if (p?.detail) {
        setError(p.detail)
        setProfile(null)
      } else {
        setProfile(p)
      }
      if (Array.isArray(its)) setItems(its)
    } catch (e: unknown) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
  }, [workspaceId, statusFilter])

  async function rate(itemId: string, rating: number) {
    await api.recommenderRate(workspaceId, itemId, rating)
    refresh()
  }

  async function materializeItem(itemId: string) {
    setMaterializing(itemId)
    try {
      await api.recommenderMaterialize(workspaceId, [itemId])
      refresh()
    } catch (e) {
      setError(String(e))
    } finally {
      setMaterializing(null)
    }
  }

  function copyMoreCommand() {
    const cmd = `loom: feed_more workspace_id="${workspaceId}" n=10 window="1w"`
    navigator.clipboard?.writeText(cmd)
  }

  if (loading && !profile) {
    return <div style={{ padding: 16 }}>Loading recommender…</div>
  }
  if (error) {
    return (
      <div style={{ padding: 16 }}>
        <p style={{ color: '#c33' }}>{error}</p>
        <p style={{ fontSize: 13, color: '#666' }}>
          This workspace may not have a recommender attached, or its profile hasn't been initialized.
          Run <code>feed_create</code> from Claude Code to set one up.
        </p>
      </div>
    )
  }

  return (
    <div style={{ padding: 16, fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
        <div>
          <h2 style={{ margin: 0 }}>{workspaceId}</h2>
          <div style={{ fontSize: 13, color: '#666', marginTop: 4 }}>
            Stage {profile?.ranker_stage ?? 0} · {profile?.pos_count ?? 0} pos / {profile?.neg_count ?? 0} neg ·{' '}
            seeds: {(profile?.seed_topics ?? []).join(', ') || '—'}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)} style={{ fontSize: 13 }}>
            <option value="">all</option>
            <option value="surfaced">surfaced</option>
            <option value="read">read</option>
            <option value="saved">saved</option>
            <option value="skipped">skipped</option>
            <option value="materialized">materialized</option>
          </select>
          <button
            onClick={copyMoreCommand}
            title="Copy MCP command for Claude Code"
            style={{ fontSize: 13, padding: '4px 10px', cursor: 'pointer' }}
          >
            Copy `feed_more` for Claude Code
          </button>
        </div>
      </div>

      {profile?.description && (
        <p style={{ fontSize: 13, color: '#444', marginBottom: 12, fontStyle: 'italic' }}>
          {profile.description}
        </p>
      )}

      {items.length === 0 ? (
        <div style={{ color: '#888', padding: 24, textAlign: 'center' }}>
          No items. Run <code>feed_more</code> from Claude Code to surface some.
        </div>
      ) : (
        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
          {items.map(item => (
            <li
              key={item.id}
              style={{
                border: '1px solid #ddd',
                borderRadius: 6,
                padding: 12,
                marginBottom: 10,
                background: item.exploration ? '#fffbe6' : 'white',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <a href={item.url} target="_blank" rel="noopener noreferrer" style={{ fontWeight: 600, color: '#0366d6' }}>
                    {item.title}
                  </a>
                  <div style={{ fontSize: 12, color: '#666', marginTop: 2 }}>
                    {item.kind} · {item.source}
                    {item.authors.length > 0 && ` · ${item.authors.slice(0, 3).join(', ')}`}
                    {item.published_at && ` · ${item.published_at.slice(0, 10)}`}
                    {item.exploration && (
                      <span style={{ marginLeft: 8, color: '#b08800', fontWeight: 600 }}>EXPLORE</span>
                    )}
                  </div>
                  {item.abstract && (
                    <p style={{ fontSize: 13, color: '#333', marginTop: 6, marginBottom: 0 }}>
                      {item.abstract.slice(0, 280)}
                      {item.abstract.length > 280 && '…'}
                    </p>
                  )}
                </div>
                <div style={{ minWidth: 160, textAlign: 'right' }}>
                  <div style={{ fontSize: 12, color: '#666' }}>
                    score: {item.final_score.toFixed(3)}
                  </div>
                  {item.calibrated_prob > 0 && (
                    <div style={{ fontSize: 12, color: '#666' }}>
                      P(like): {(item.calibrated_prob * 100).toFixed(0)}%
                    </div>
                  )}
                  <div style={{ fontSize: 12, color: '#666', marginBottom: 6 }}>
                    status: {item.status}
                  </div>
                  <div style={{ display: 'inline-flex', gap: 2, marginBottom: 6 }}>
                    {[1, 2, 3, 4, 5].map(n => (
                      <button
                        key={n}
                        onClick={() => rate(item.id, n)}
                        title={`Rate ${n}`}
                        style={{
                          fontSize: 12,
                          width: 26,
                          padding: '2px 0',
                          cursor: 'pointer',
                          border: '1px solid #ccc',
                          background: 'white',
                        }}
                      >
                        {n}
                      </button>
                    ))}
                  </div>
                  {item.status !== 'materialized' && (
                    <button
                      onClick={() => materializeItem(item.id)}
                      disabled={materializing === item.id}
                      title="Promote to a workspace document"
                      style={{
                        fontSize: 11,
                        padding: '2px 8px',
                        cursor: 'pointer',
                        border: '1px solid #0366d6',
                        background: 'white',
                        color: '#0366d6',
                        borderRadius: 3,
                      }}
                    >
                      {materializing === item.id ? '…' : 'Materialize'}
                    </button>
                  )}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
