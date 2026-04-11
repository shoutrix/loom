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

  readPaper: (identifier: string) =>
    post('/papers/read', { identifier }),

  paperContent: (paperId: string) =>
    get(`/papers/${encodeURIComponent(paperId)}/content`),

  queuePapers: (paperIds: string[] = [], identifiers: string[] = []) =>
    post('/papers/queue', { paper_ids: paperIds, identifiers }),

  queueStatus: () => get('/papers/queue/status'),

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

  health: () => get('/health'),
}
