import { useState, useEffect, useRef, useCallback } from 'react'
import { api } from '../api/client'
import { RefreshCw, Loader2, X } from 'lucide-react'

interface GraphEntity {
  id: string
  name: string
  type: string
  description: string
  mentions: number
  community_id: string
  confidence: number
  source_doc_ids: string[]
}

interface GraphRelationship {
  source_entity: string
  target_entity: string
  type: string
  description: string
}

interface GraphData {
  nodes: { id: string; name: string; type: string; val: number; community: string; color: string }[]
  links: { source: string; target: string; type: string }[]
}

const TYPE_COLORS: Record<string, string> = {
  concept: '#3b82f6',
  technique: '#ef4444',
  method: '#f97316',
  system: '#8b5cf6',
  paper: '#22c55e',
  dataset: '#eab308',
  metric: '#06b6d4',
  claim: '#a855f7',
}

export function GraphView() {
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<GraphEntity | null>(null)
  const [ForceGraph, setForceGraph] = useState<any>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 380, height: 600 })

  useEffect(() => {
    import('react-force-graph-3d').then(mod => setForceGraph(() => mod.default))
  }, [])

  const loadGraph = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.graphExport()
      const entities: GraphEntity[] = data.entities || []
      const relationships: GraphRelationship[] = data.relationships || []

      const nodes = entities.map(e => ({
        id: e.id,
        name: e.name,
        type: e.type,
        val: Math.max(2, Math.min(10, e.mentions * 2)),
        community: e.community_id || '',
        color: TYPE_COLORS[e.type] || '#9ca3af',
      }))

      const nodeIds = new Set(nodes.map(n => n.id))
      const links = relationships
        .filter(r => nodeIds.has(r.source_entity) && nodeIds.has(r.target_entity))
        .map(r => ({
          source: r.source_entity,
          target: r.target_entity,
          type: r.type,
        }))

      setGraphData({ nodes, links })
    } catch (e) {
      console.error(e)
    }
    setLoading(false)
  }, [])

  useEffect(() => { loadGraph() }, [loadGraph])

  useEffect(() => {
    if (!containerRef.current) return
    const obs = new ResizeObserver(entries => {
      for (const entry of entries) {
        setDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        })
      }
    })
    obs.observe(containerRef.current)
    return () => obs.disconnect()
  }, [])

  const handleNodeClick = useCallback(async (node: any) => {
    try {
      const data = await api.graphEntities(1000)
      const entities: GraphEntity[] = data.entities || []
      const ent = entities.find((e: GraphEntity) => e.id === node.id)
      setSelected(ent || null)
    } catch {
      setSelected(null)
    }
  }, [])

  return (
    <div className="flex flex-col h-full relative">
      <div className="flex items-center justify-between px-3 py-2 border-b border-surface-3 shrink-0">
        <span className="text-sm font-medium text-text-primary">Knowledge Graph</span>
        <div className="flex items-center gap-2">
          {graphData && (
            <span className="text-xs text-text-muted">
              {graphData.nodes.length} nodes
            </span>
          )}
          <button onClick={loadGraph} className="text-text-muted hover:text-text-secondary transition-colors">
            <RefreshCw size={13} />
          </button>
        </div>
      </div>

      <div ref={containerRef} className="flex-1 relative overflow-hidden">
        {loading ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <Loader2 size={24} className="animate-spin text-text-muted" />
          </div>
        ) : graphData && ForceGraph ? (
          <ForceGraph
            graphData={graphData}
            width={dimensions.width}
            height={dimensions.height}
            backgroundColor="#ffffff"
            nodeLabel={(node: any) => `${node.name} (${node.type})`}
            nodeColor={(node: any) => node.color}
            nodeRelSize={4}
            nodeOpacity={0.9}
            linkColor={() => '#d1d5db'}
            linkWidth={0.5}
            linkOpacity={0.4}
            onNodeClick={handleNodeClick}
            cooldownTicks={100}
            d3AlphaDecay={0.02}
            d3VelocityDecay={0.3}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-text-muted text-sm">
            No graph data
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="px-3 py-2 border-t border-surface-3 flex flex-wrap gap-2 shrink-0">
        {Object.entries(TYPE_COLORS).map(([type, color]) => (
          <span key={type} className="flex items-center gap-1 text-[10px] text-text-muted">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
            {type}
          </span>
        ))}
      </div>

      {/* Selected entity */}
      {selected && (
        <div className="absolute bottom-10 left-2 right-2 bg-white border border-surface-3 rounded-lg shadow-lg p-3 max-h-48 overflow-y-auto">
          <div className="flex items-start justify-between mb-2">
            <div>
              <h4 className="text-sm font-medium text-text-primary">{selected.name}</h4>
              <span
                className="text-xs px-1.5 py-0.5 rounded mt-0.5 inline-block font-medium"
                style={{
                  backgroundColor: (TYPE_COLORS[selected.type] || '#9ca3af') + '15',
                  color: TYPE_COLORS[selected.type] || '#9ca3af',
                }}
              >
                {selected.type}
              </span>
            </div>
            <button onClick={() => setSelected(null)} className="text-text-muted hover:text-text-primary">
              <X size={14} />
            </button>
          </div>
          <p className="text-xs text-text-secondary">{selected.description}</p>
          <div className="flex gap-3 mt-2 text-xs text-text-muted">
            <span>Mentions: {selected.mentions}</span>
            <span>Confidence: {(selected.confidence * 100).toFixed(0)}%</span>
            <span>Sources: {selected.source_doc_ids?.length || 0}</span>
          </div>
        </div>
      )}
    </div>
  )
}
