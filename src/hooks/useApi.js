import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client.js'

// Fetch a GET endpoint; returns { data, error, loading, reload }.
export function useApi(path, deps = []) {
  const [state, set] = useState({ data: null, error: null, loading: !!path })
  const alive = useRef(true)
  const load = useCallback(async () => {
    if (!path) return
    set((s) => ({ ...s, loading: true }))
    try {
      const data = await api.get(path)
      if (alive.current) set({ data, error: null, loading: false })
    } catch (error) {
      if (alive.current) set({ data: null, error, loading: false })
    }
  }, [path])
  useEffect(() => { alive.current = true; load(); return () => { alive.current = false } }, [load, ...deps])
  return { ...state, reload: load }
}

// Poll a background job until it finishes. Every state shown comes from the backend.
export function useJob(jobId, onDone) {
  const [job, setJob] = useState(null)
  const doneRef = useRef(onDone)
  doneRef.current = onDone
  useEffect(() => {
    if (!jobId) { setJob(null); return }
    let stop = false
    let timer
    const tick = async () => {
      try {
        const j = await api.get(`/api/jobs/${jobId}`)
        if (stop) return
        setJob(j)
        if (['completed', 'failed', 'cancelled'].includes(j.status)) { doneRef.current?.(j); return }
      } catch (e) { /* transient; keep polling */ }
      timer = setTimeout(tick, 900)
    }
    tick()
    return () => { stop = true; clearTimeout(timer) }
  }, [jobId])
  return job
}

// Re-fetch every `ms` while `active` is true.
export function useInterval(fn, ms, active = true) {
  const ref = useRef(fn)
  ref.current = fn
  useEffect(() => {
    if (!active) return
    const t = setInterval(() => ref.current(), ms)
    return () => clearInterval(t)
  }, [ms, active])
}
