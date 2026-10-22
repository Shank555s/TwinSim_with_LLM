import { useEffect, useState } from 'react'
import { api } from '../api/client.js'
import { Btn, ErrorNote, Field, FailureTags, Page, Panel, Tag } from '../components/ui.jsx'
import { AXES, fmt, signed } from '../constants.js'
import Pipeline, { FULL_PIPELINE } from '../experiment/Pipeline.jsx'
import { useJob } from '../hooks/useApi.js'
import { useLab } from '../state.jsx'

const DEMO_STEPS = [...FULL_PIPELINE, { k: 'rerun', l: 'Same scenario again' }]

function Why({ ev }) {
  if (!ev) return null
  const label = (v) => (v >= 8 ? ['strong', 'var(--good)'] : v >= 6 ? ['moderate', 'var(--warn)'] : ['weak', 'var(--bad)'])
  return (
    <div className="stack" style={{ gap: 3 }}>
      {AXES.map((a) => { const [w, c] = label(ev.axes[a.k]); return <div key={a.k} className="row" style={{ justifyContent: 'space-between', fontSize: 13 }}><span className="sub">{a.l}</span><span style={{ color: c }}>{w} {fmt(ev.axes[a.k])}</span></div> })}
      <div style={{ marginTop: 6 }}><FailureTags codes={ev.failure_codes} /></div>
    </div>
  )
}

function Side({ title, model, block }) {
  return (
    <Panel title={title} hint={model}>
      {!block ? <p className="faint">Waiting…</p> : (
        <div className="stack">
          <div style={{ fontFamily: 'var(--serif)', fontSize: 52, lineHeight: 1 }}>{fmt(block.evaluation.overall)}<span className="faint" style={{ fontSize: 14, fontFamily: 'var(--sans)' }}> fidelity</span></div>
          <div className={`response small ${block.response ? '' : 'empty'}`}>{block.response || '(no text)'}</div>
          <Why ev={block.evaluation} />
        </div>
      )}
    </Panel>
  )
}

export default function Demo({ params }) {
  const lab = useLab()
  const [pid, setPid] = useState('')
  const [sid, setSid] = useState('')
  const [jobId, setJobId] = useState(params?.job || null)
  const [error, setError] = useState(null)
  useEffect(() => {
    if (!pid && lab.personas.length) setPid(lab.personas.find((p) => !p.is_benchmark)?.id || lab.personas[0].id)
    if (!sid && lab.scenarios.length) setSid(lab.scenarios.find((s) => s.split === 'train' && s.domain === 'political')?.id || lab.scenarios[0].id)
  }, [lab.personas, lab.scenarios])
  const job = useJob(jobId, () => lab.refresh())
  useEffect(() => { // show the persona/scenario the job actually used
    if (job?.payload?.persona_id) { setPid(job.payload.persona_id); setSid(job.payload.scenario_id) }
  }, [job?.id])
  const running = job && ['queued', 'running'].includes(job.status)
  const firstEv = job?.events?.find((e) => e.data?.first)?.data.first
  const r = job?.status === 'completed' ? job.result : null
  const start = async () => {
    setError(null)
    try { setJobId((await api.post('/api/demo/run', { persona_id: pid, scenario_id: sid, seed: Math.floor(Math.random() * 1000) })).job_id) } catch (e) { setError(e) }
  }
  const persona = lab.personas.find((p) => p.id === pid)
  return (
    <Page title="Demo" intro="The whole idea in one run: the twin answers, gets scored, learns from feedback, becomes a new model version and answers the same scenario again with the same random seed. Nothing here is scripted. On a CPU it takes a few minutes.">
      <Panel>
        <div className="row" style={{ alignItems: 'flex-end', gap: 14 }}>
          <Field label="Persona"><select value={pid} onChange={(e) => setPid(e.target.value)} style={{ width: 260 }}>{lab.personas.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}</select></Field>
          <Field label="Scenario"><select value={sid} onChange={(e) => setSid(e.target.value)} style={{ width: 420 }}>{lab.scenarios.map((s) => <option key={s.id} value={s.id}>[{s.domain}] {s.text.slice(0, 80)}</option>)}</select></Field>
          <Btn kind="primary" onClick={start} disabled={running || !lab.active}>Run demo</Btn>
        </div>
        {persona?.is_benchmark && <p className="faint" style={{ marginTop: 8 }}>Benchmark personas are protected: this run's own feedback won't be trained on, but the model still learns from harvested training prompts.</p>}
      </Panel>
      <ErrorNote error={error} />
      {job && <Panel title="Pipeline"><Pipeline job={job} steps={DEMO_STEPS} /></Panel>}
      {job && (
        <div className="grid g2" style={{ alignItems: 'start' }}>
          <Side title="Before" model={r?.before_model || lab.active} block={r?.first || firstEv} />
          <Side title="After training" model={r?.after_model} block={r?.second} />
        </div>
      )}
      {r && (
        <div className={`banner ${r.accepted ? 'good' : 'bad'}`}>
          <div className="stack" style={{ gap: 4 }}>
            <strong>
              Same scenario, same seed: fidelity {fmt(r.first.evaluation.overall)} → {fmt(r.second.evaluation.overall)} ({signed(r.improvement)}).{' '}
              {r.accepted ? `${r.after_model} passed the benchmark and is now active.` : `${r.after_model} did not pass the benchmark, so ${r.before_model} stays active.`}
            </strong>
            <span className="sub">Benchmark fidelity {fmt(r.decision.previous_score)} → {fmt(r.decision.new_score)} ({signed(r.decision.improvement, 2)}). One scenario can move either way by chance; the benchmark decides.</span>
            <span className="faint">Harvest: {r.harvest?.examples_added} new examples from {r.harvest?.samples} samples · training {r.training.steps} steps, validation loss {fmt(r.training.val_loss, 3)}</span>
          </div>
          <Tag kind={r.accepted ? 'good' : 'bad'}>{r.accepted ? 'promoted' : 'rejected'}</Tag>
        </div>
      )}
    </Page>
  )
}
