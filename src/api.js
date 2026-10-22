// Thin client for the TwinSim FastAPI backend. Every number on screen comes from these calls.
import { useCallback, useEffect, useRef, useState } from 'react'

export class ApiError extends Error {
  constructor(status, detail) {
    const msg = typeof detail === 'string' ? detail : detail?.message || (Array.isArray(detail) ? detail.map(d => d.msg).join('; ') : 'request failed')
    super(msg)
    this.status = status
    this.code = detail?.code
  }
}

export async function api(path, { method = 'GET', body, signal } = {}) {
  let res
  try {
    res = await fetch(`/api${path}`, {
      method, signal,
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
  } catch (e) {
    if (e.name === 'AbortError') throw e
    throw new ApiError(0, { code: 'offline', message: 'Backend unreachable. Is `uvicorn main:app --port 8000` running in backend/?' })
  }
  const text = await res.text()
  let data = null
  try { data = text ? JSON.parse(text) : null } catch { data = text }
  if (!res.ok) throw new ApiError(res.status, data?.detail ?? data)
  return data
}

export function useApi(path, { poll = 0, enabled = true } = {}) {
  const [state, setState] = useState({ data: null, error: null, loading: !!path && enabled })
  const tick = useRef(0)
  const load = useCallback(async (quiet = false) => {
    if (!path || !enabled) return
    const id = ++tick.current
    if (!quiet) setState(s => ({ ...s, loading: true }))
    try {
      const data = await api(path)
      if (id === tick.current) setState({ data, error: null, loading: false })
    } catch (error) {
      if (id === tick.current) setState(s => ({ data: s.data, error, loading: false }))
    }
  }, [path, enabled])
  useEffect(() => { load() }, [load])
  useEffect(() => {
    if (!poll || !path || !enabled) return
    const t = setInterval(() => { if (!document.hidden) load(true) }, poll)
    return () => clearInterval(t)
  }, [poll, path, enabled, load])
  return { ...state, reload: load }
}

const TERMINAL = new Set(['completed', 'failed', 'cancelled'])
export const isTerminal = s => TERMINAL.has(s)

// Poll a background job until it finishes. The job row is the backend's own record of progress.
export function useJob(jobId, { interval = 800, onDone } = {}) {
  const [job, setJob] = useState(null)
  const done = useRef(onDone)
  done.current = onDone
  useEffect(() => {
    setJob(null)
    if (!jobId) return
    let alive = true, timer
    const step = async () => {
      try {
        const j = await api(`/jobs/${jobId}`)
        if (!alive) return
        setJob(j)
        if (isTerminal(j.status)) { done.current?.(j); return }
      } catch { /* transient; retry */ }
      timer = setTimeout(step, interval)
    }
    step()
    return () => { alive = false; clearTimeout(timer) }
  }, [jobId, interval])
  return job
}

// POST /experiments/trial/stream: server-sent events of the real pipeline stages and tokens.
export async function streamTrial(body, onEvent, signal) {
  let res
  try {
    res = await fetch('/api/experiments/trial/stream', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal,
    })
  } catch (e) {
    if (e.name === 'AbortError') throw e
    throw new ApiError(0, { message: 'Backend unreachable' })
  }
  if (!res.ok) {
    let d = null
    try { d = (await res.json()).detail } catch { /* ignore */ }
    throw new ApiError(res.status, d)
  }
  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let i
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, i)
      buf = buf.slice(i + 2)
      for (const line of chunk.split('\n')) {
        if (line.startsWith('data:')) {
          try { onEvent(JSON.parse(line.slice(5))) } catch { /* malformed event */ }
        }
      }
    }
  }
}

// ------------------------------------------------------------------ formatting
export const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? '–' : Number(v).toFixed(d))
export const signed = (v, d = 1) => (v === null || v === undefined ? '–' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(d)}`)
export const pct = (v, d = 0) => (v === null || v === undefined ? '–' : `${(v * 100).toFixed(d)}%`)
export const int = v => (v === null || v === undefined ? '–' : Number(v).toLocaleString('en-US'))
export function compact(n) {
  if (n === null || n === undefined) return '–'
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return String(n)
}
export function timeAgo(ts) {
  if (!ts) return '–'
  const s = Date.now() / 1000 - ts
  if (s < 45) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)}m ago`
  if (s < 86400) return `${Math.round(s / 3600)}h ago`
  return `${Math.round(s / 86400)}d ago`
}
export const clock = ts => (ts ? new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '–')
export const dateTime = ts => (ts ? new Date(ts * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '–')
export function duration(sec) {
  if (sec === null || sec === undefined) return '–'
  if (sec < 60) return `${sec.toFixed(0)}s`
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`
  return `${Math.floor(sec / 3600)}h ${Math.round((sec % 3600) / 60)}m`
}
export const shortModel = n => (n ? n.replace('TwinSim-', '') : '–')

export const TRAITS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
export const AXES = [...TRAITS, 'political', 'style']
export const AXIS_LABEL = {
  openness: 'Openness', conscientiousness: 'Conscientiousness', extraversion: 'Extraversion',
  agreeableness: 'Agreeableness', neuroticism: 'Neuroticism', political: 'Politics', style: 'Communication',
}
export const DOMAIN_META = {
  economic: { icon: 'coins', color: '#D9A441', label: 'Economic Decision', blurb: 'Prices, money, trade-offs' },
  political: { icon: 'landmark', color: '#D99A6C', label: 'Political Opinion', blurb: 'Policy and ideology' },
  health: { icon: 'heart', color: '#5FB37A', label: 'Health Choice', blurb: 'Medical and lifestyle choices' },
  social: { icon: 'chat', color: '#7FA7B8', label: 'Social Media Reply', blurb: 'Short-form public replies' },
}
export const STATUS_TONE = { active: 'violet', baseline: 'cyan', archived: '', rejected: 'red', candidate: 'amber' }
export const fidelityTone = v => (v === null || v === undefined ? '' : v >= 75 ? 'green' : v >= 55 ? 'amber' : 'red')
export const fidelityColor = v => (v === null || v === undefined ? '#7A7168' : v >= 75 ? '#5FB37A' : v >= 55 ? '#D9A441' : '#D96459')
