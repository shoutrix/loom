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

async function put(path: string, body?: unknown) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  return res.json()
}

async function del(path: string) {
  const res = await fetch(`${BASE}${path}`, { method: 'DELETE' })
  return res.json()
}

export type DocType =
  | 'research_paper'
  | 'article'
  | 'note'
  | 'transcript'
  | 'spec'
  | 'memo'
  | 'book_chapter'
  | 'documentation'

export interface DocumentDescriptor {
  doc_id: string
  doc_type: DocType
  title: string
  body_path?: string
  source_url?: string
  authors?: string[]
  published_at?: string
  references?: { title?: string; url?: string; arxiv_id?: string; doc_id?: string }[]
  tldr?: string
  category_path?: string[]
  metadata_status?: 'pending' | 'deriving' | 'derived' | 'failed'
  created_at?: string
  updated_at?: string
}

export interface SubmitDocumentRequest {
  body: string
  doc_type?: DocType
  title?: string
  source_url?: string
  authors?: string[]
  published_at?: string
  references?: { title?: string; url?: string; arxiv_id?: string }[]
  category_path?: string[]
}

export const api = {
  // ----- documents ---------------------------------------------------
  documents: () => get('/documents'),
  documentsContents: () => get('/documents/contents'),
  document: (docId: string) => get(`/documents/${encodeURIComponent(docId)}`),
  documentBody: (docId: string) => get(`/documents/${encodeURIComponent(docId)}/body`),
  submitDocument: (req: SubmitDocumentRequest) => post('/documents', req),
  deleteDocument: (docId: string) => del(`/documents/${encodeURIComponent(docId)}`),
  queueStatus: () => get('/documents/queue/status'),

  // ----- workspaces --------------------------------------------------
  workspaces: () => get('/workspaces'),
  activeWorkspace: () => get('/workspaces/active'),
  renameWorkspace: (name: string) => patch('/workspaces/active/name', { name }),
  createWorkspace: (workspaceId: string, description = '') =>
    post('/workspaces/create', { workspace_id: workspaceId, description }),
  switchWorkspace: (workspaceId: string) =>
    post('/workspaces/switch', { workspace_id: workspaceId }),
  deleteWorkspace: (workspaceId: string) => del(`/workspaces/${workspaceId}`),

  // ----- workspace brief ---------------------------------------------
  workspaceBrief: () => get('/workspaces/brief'),
  setWorkspaceBrief: (body: { brief?: any; user_notes?: string }) =>
    put('/workspaces/brief', body),
  regenerateWorkspaceBrief: () => post('/workspaces/brief/regenerate'),

  // ----- chat --------------------------------------------------------
  chat: (message: string) => post('/chat', { message }),
  clearChat: () => post('/chat/clear'),

  // ----- graph -------------------------------------------------------
  graphStats: () => get('/graph/stats'),
  graphEntities: (limit = 500) => get(`/graph/entities?limit=${limit}`),
  graphCommunities: () => get('/graph/communities'),
  graphExport: () => get('/graph/export'),

  // ----- recommender (optional capability) ---------------------------
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

  health: () => get('/health'),
}
