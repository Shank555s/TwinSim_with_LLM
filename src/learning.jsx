// Automatic learning is an internal process: nothing here starts, stops or configures it.
//   LearningPill  small read-only status in the top bar ("Learning update in progress…", "Model updated", …)
//   Bootstrap     one-time creation of the baseline model (TwinSim-v0) on a fresh install; not a learning action
import { useEffect, useRef, useState } from 'react'
import Icon from './icons.jsx'
import { api, compact, fmt, pct, signed, useApi, useJob } from './api.js'
import { Btn, Card, CardHead, Dot, ErrorBox, Progress, Seg, useApp } from './ui.jsx'

const TONE = { learning: 'violet', updated: 'green', completed: 'cyan', collecting: 'amber', idle: '', disabled: '' }

export function LearningPill({ model }) {
  const { toast } = useApp()
  const st = useApi('/learning/status', { poll: 4000 })
  const d = st.data
  const prev = useRef(null)
  useEffect(() => {
    if (!d) return
    if (prev.current === 'learning' && (d.state === 'updated' || d.state === 'completed')) toast(d.message, d.state === 'updated' ? 'ok' : '')
    prev.current = d.state
  }, [d?.state]) // eslint-disable-line react-hooks/exhaustive-deps
  if (!d || !d.active_model) return null
  const imp = d.improvement, last = d.last_cycle
  const tip = [
    d.message,
    d.state === 'collecting' ? `Learning starts after ${d.trigger_at} experiments (${d.pending_signals} queued).` : null,
    imp?.fidelity_delta !== null && imp?.fidelity_delta !== undefined ? `Measured on the fixed benchmark: ${fmt(imp.baseline?.fidelity, 1)} → ${fmt(imp.active?.fidelity, 1)} (${signed(imp.fidelity_delta, 1)}), failure rate ${pct(imp.baseline?.failure_rate)} → ${pct(imp.active?.failure_rate)}.` : null,
    last ? `Last cycle: ${last.model_after || '—'} ${last.accepted ? 'promoted' : last.status === 'failed' ? 'failed' : 'not promoted'} (${last.n_signals} experiments).` : null,
    `${d.experiments_incorporated} experiments learned from.`,
  ].filter(Boolean).join('\n')
  return (
    <span className="top-pill hide-sm" title={tip} aria-live="polite">
      <Dot tone={TONE[d.state]} pulse={d.state === 'learning'} />
      <span>{d.state === 'idle' ? 'Learning up to date' : d.message}</span>
      {d.state === 'learning' && d.current?.progress != null && <span className="dim mono">{Math.round(d.current.progress * 100)}%</span>}
    </span>
  )
}

export function Bootstrap({ onDone }) {
  const sys = useApi('/system')
  const active = useApi('/jobs?active=true&limit=5', { poll: 2500 })
  const { refreshStatus, status } = useApp()
  const [profile, setProfile] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [err, setErr] = useState(null)
  const rec = sys.data?.recommended
  useEffect(() => { if (rec && !profile) setProfile(rec.profile) }, [rec]) // eslint-disable-line react-hooks/exhaustive-deps
  const running = active.data?.jobs?.find(j => j.type === 'bootstrap')
  useEffect(() => { if (running && running.id !== jobId) setJobId(running.id) }, [running?.id]) // eslint-disable-line react-hooks/exhaustive-deps
  const job = useJob(jobId, { interval: 1500, onDone: () => { refreshStatus(); onDone?.() } })
  const busy = job && ['queued', 'running'].includes(job.status)
  async function go() {
    setErr(null)
    try { const r = await api('/training/bootstrap', { method: 'POST', body: { profile } }); setJobId(r.job_id); refreshStatus() } catch (e) { setErr(e) }
  }
  return (
    <Card glow="v">
      <CardHead icon="bolt" title="Set up the baseline twin model" sub="One-time: tokenizer → language pre-training → persona specialisation → fixed benchmark. After this the model improves by itself as you run experiments." />
      {status?.public_mode ? <p className="small muted">The baseline is created on the server with <span className="mono">scripts/train.py bootstrap</span>.</p> : (<>
        <div className="row wrap gap-16">
          <Seg value={profile} onChange={setProfile} options={[['tiny', 'tiny'], ['dev', 'dev'], ['research', 'research'], ['research-large', 'research-large']]} />
          <Btn kind="primary" icon="play" disabled={!profile || busy} onClick={go}>Train baseline model</Btn>
        </div>
        {rec && <p className="small muted mt-8">Recommended for this machine: <b className="violet">{rec.profile}</b> ({rec.reason}) · ~{compact(rec.approx_parameters)} parameters.</p>}
      </>)}
      {job && (
        <div className="mt-12">
          <div className="row between small"><span className="muted">{job.stage}: {job.message}</span><span className="mono">{Math.round((job.progress || 0) * 100)}%</span></div>
          <div className="mt-8"><Progress value={job.progress} /></div>
          {job.status === 'failed' && <div className="mt-12"><ErrorBox>{job.error?.split('\n')[0]}</ErrorBox></div>}
        </div>
      )}
      <ErrorBox error={err} />
    </Card>
  )
}
