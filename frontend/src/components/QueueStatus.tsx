import { useState, useEffect } from 'react'
import { api } from '../api/client'
import { Loader2, CheckCircle } from 'lucide-react'

export function QueueStatus() {
  const [status, setStatus] = useState<{
    running: boolean
    queue_depth: number
    current_paper: string | null
  } | null>(null)

  useEffect(() => {
    const poll = async () => {
      try {
        const data = await api.queueStatus()
        setStatus(data)
      } catch {
        // ignore
      }
    }
    poll()
    const interval = setInterval(poll, 3000)
    return () => clearInterval(interval)
  }, [])

  if (!status) return null

  const isActive = status.queue_depth > 0 || status.current_paper

  if (!isActive) {
    return (
      <div className="flex items-center gap-1.5 text-xs text-text-muted">
        <CheckCircle size={12} className="text-emerald-500" />
        <span>Idle</span>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-1.5 text-xs text-accent bg-accent/5 px-2.5 py-1 rounded-lg border border-accent/10">
      <Loader2 size={12} className="animate-spin" />
      <span className="max-w-[180px] truncate">
        {status.current_paper
          ? `Ingesting: ${status.current_paper}`
          : `${status.queue_depth} in queue`}
      </span>
      {status.queue_depth > 0 && status.current_paper && (
        <span className="text-text-muted">+{status.queue_depth}</span>
      )}
    </div>
  )
}
