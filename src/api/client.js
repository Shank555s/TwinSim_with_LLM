// Thin client for the TwinSim FastAPI backend. No API keys: the twin runs on our own model.
const BASE = import.meta.env?.VITE_API_BASE || ''

export class ApiError extends Error {
  constructor(status, detail) {
    const msg = typeof detail === 'string' ? detail : detail?.message || JSON.stringify(detail)
    super(msg)
    this.status = status
    this.code = detail?.code
  }
}

async function req(method, path, body) {
  let r
  try {
    r = await fetch(BASE + path, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    })
  } catch (e) {
    throw new ApiError(0, { code: 'offline', message: 'Cannot reach the TwinSim backend. Start it with: cd backend && uvicorn main:app --port 8000' })
  }
  const text = await r.text()
  let data = null
  try { data = text ? JSON.parse(text) : null } catch { data = text }
  if (!r.ok) throw new ApiError(r.status, data?.detail ?? data)
  return data
}

export const api = {
  get: (p) => req('GET', p),
  post: (p, b = {}) => req('POST', p, b),
  put: (p, b) => req('PUT', p, b),
  del: (p) => req('DELETE', p),
  url: (p) => BASE + p,
}
