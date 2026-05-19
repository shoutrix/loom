const BASE = ''

async function post(path: string, body?: unknown) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  return res.json()
}

async function get(path: string) {
  const res = await fetch(`${BASE}${path}`)
  return res.json()
}

async function patch(path: string, body?: unknown) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  return res.json()
}

async function del(path: string) {
  const res = await fetch(`${BASE}${path}`, { method: 'DELETE' })
  return res.json()
}

export const api = {
  searchPapers: (query: string, maxResults = 20) =>
    post('/papers/search', { query, max_results: maxResults, enable_graph_expansion: true }),

  startSearch: (query: string, maxResults = 20) =>
    post('/papers/search/start', { query, max_results: maxResults, enable_graph_expansion: true }),

  searchStatus: (searchId: string) =>
    get(`/papers/search/${encodeURIComponent(searchId)}/status`),

  searchResult: (searchId: string) =>
    get(`/papers/search/${encodeURIComponent(searchId)}/result`),

  stopSearch: (searchId: string) =>
    post(`/papers/search/${encodeURIComponent(searchId)}/stop`),

  readPaper: (identifier: string) =>
    post('/papers/read', { identifier }),

  paperContent: (paperId: string) =>
    get(`/papers/${encodeURIComponent(paperId)}/content`),

  queuePapers: (paperIds: string[] = [], identifiers: string[] = []) =>
    post('/papers/queue', { paper_ids: paperIds, identifiers }),

  queueStatus: () => get('/papers/queue/status'),

  exploreGraph: (paperId: string, title: string, abstract: string) =>
    post('/papers/explore-graph/start', { paper_id: paperId, title, abstract }),

  exploreGraphStatus: (jobId: string) =>
    get(`/papers/explore-graph/${encodeURIComponent(jobId)}/status`),

  exploreGraphResult: (jobId: string) =>
    get(`/papers/explore-graph/${encodeURIComponent(jobId)}/result`),

  registry: () => get('/papers/registry'),

  chat: (message: string) => post('/chat', { message }),

  clearChat: () => post('/chat/clear'),

  graphStats: () => get('/graph/stats'),

  graphEntities: (limit = 500) => get(`/graph/entities?limit=${limit}`),

  graphCommunities: () => get('/graph/communities'),

  graphExport: () => get('/graph/export'),

  vaultFiles: () => get('/vault/files'),

  vaultRead: (path: string) => get(`/vault/read?path=${encodeURIComponent(path)}`),

  vaultWrite: (path: string, content: string) =>
    post('/vault/write', { path, content }),

  vaultNote: (title: string, content: string) =>
    post('/vault/note', { title, content }),

  ingestUrl: (url: string) => post('/ingest/url', { url }),

  ingestFile: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return fetch('/ingest/file', { method: 'POST', body: form }).then(r => r.json())
  },

  workspaces: () => get('/workspaces'),

  activeWorkspace: () => get('/workspaces/active'),

  renameWorkspace: (name: string) =>
    patch('/workspaces/active/name', { name }),

  createWorkspace: (workspaceId: string, description = '') =>
    post('/workspaces/create', { workspace_id: workspaceId, description }),

  switchWorkspace: (workspaceId: string) =>
    post('/workspaces/switch', { workspace_id: workspaceId }),

  deleteWorkspace: (workspaceId: string) =>
    del(`/workspaces/${workspaceId}`),

  // Recommender (HTTP routes under /feed — kept for path-stability while
  // package and tool names migrate to "recommender").
  recommenderProfile: (workspaceId: string) =>
    get(`/feed/profile/${encodeURIComponent(workspaceId)}`),

  recommenderItems: (
    workspaceId: string,
    opts?: { status?: string; limit?: number; runId?: string },
  ) => {
    const params = new URLSearchParams()
    if (opts?.status) params.set('status', opts.status)
    if (opts?.limit) params.set('limit', String(opts.limit))
    if (opts?.runId) params.set('run_id', opts.runId)
    const qs = params.toString()
    return get(`/feed/items/${encodeURIComponent(workspaceId)}${qs ? '?' + qs : ''}`)
  },

  recommenderRuns: (workspaceId: string, limit = 10) =>
    get(`/feed/runs/${encodeURIComponent(workspaceId)}?limit=${limit}`),

  recommenderRate: (workspaceId: string, itemId: string, rating: number, note = '') =>
    post(
      `/feed/items/${encodeURIComponent(workspaceId)}/${encodeURIComponent(itemId)}/rate`,
      { rating, note },
    ),

  // Materialize is currently only exposed via MCP; the UI button is a UX
  // affordance for the user to copy the right MCP call. Once an HTTP route
  // is added this method will hit it directly.
  recommenderMaterialize: (workspaceId: string, itemIds: string[]) => {
    const cmd = `loom: materialize_into_workspace workspace_id="${workspaceId}" item_ids=${JSON.stringify(itemIds)}`
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      navigator.clipboard.writeText(cmd)
    }
    return Promise.resolve({
      ok: true,
      hint: 'MCP command copied to clipboard. Run it from Claude Code or Cursor to materialize.',
      command: cmd,
    })
  },

  health: () => get('/health'),
}
