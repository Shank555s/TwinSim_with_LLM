import { useEffect, useMemo, useState } from 'react'
import Icon from '../icons.jsx'
import { AXIS_LABEL, DOMAIN_META, api, dateTime, fidelityColor, fidelityTone, fmt, pct, useApi, useJob } from '../api.js'
import { Badge, Btn, Card, CardHead, CountUp, Empty, ErrorBox, Field, Note, PageHead, Progress, Seg, Slider, Thinking, Toggle, useApp } from '../ui.jsx'

export default function ExperimentLab({ params }) {
  const [tab, setTab] = useState(params?.tab || 'study')
  return (<>
    <PageHead eyebrow={<><Icon name="grid" size={13} />Experiment</>} title="Experiment Lab"
      sub="Run controlled studies (N personas × N scenarios × N repetitions), or rate twin answers blind. Every experiment also feeds the automatic learner."
      right={<Seg value={tab} onChange={setTab} options={[['study', 'N×N×N study'], ['human', 'Human evaluation']]} />} />
    <div key={tab} className="fade-in">
      {tab === 'study' && <Study />}
      {tab === 'human' && <HumanEval />}
    </div>
  </>)
}

// ------------------------------------------------------------------ study
function Study() {
  const { refreshStatus, status } = useApp()
  const pub = !!status?.public_mode
  const personas = useApi('/personas')
  const scen = useApi('/scenarios')
  const past = useApi('/experiments?kind=study&limit=20')
  const [pids, setPids] = useState([])
  const [sids, setSids] = useState([])
  const [domain, setDomain] = useState('all')
  const [reps, setReps] = useState(2)
  const [temp, setTemp] = useState(0.7)
  const [model, setModel] = useState('active')
  const [name, setName] = useState('Study')
  const [jobId, setJobId] = useState(null)
  const [err, setErr] = useState(null)
  const [openExp, setOpenExp] = useState(null)
  const job = useJob(jobId, { onDone: j => { past.reload(true); refreshStatus(); if (j.result?.experiment_id) setOpenExp(j.result.experiment_id) } })

  useEffect(() => {
    if (personas.data && !pids.length) setPids(personas.data.personas.filter(p => p.is_benchmark).slice(0, 3).map(p => p.id))
  }, [personas.data]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (scen.data && !sids.length) setSids(scen.data.scenarios.filter(s => s.split === 'benchmark').slice(0, 3).map(s => s.id))
  }, [scen.data]) // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = (arr, set, id) => set(arr.includes(id) ? arr.filter(x => x !== id) : [...arr, id])
  const shown = (scen.data?.scenarios || []).filter(s => domain === 'all' || s.domain === domain)
  const total = pids.length * sids.length * reps
  const running = job && ['queued', 'running'].includes(job.status)

  async function start() {
    setErr(null)
    try {
      const r = await api('/experiments/study', { method: 'POST', body: { persona_ids: pids, scenario_ids: sids, repetitions: reps, model_version: model, name, generation: { temperature: temp, seed: 1 } } })
      setJobId(r.job_id); refreshStatus()
    } catch (e) { setErr(e) }
  }

  return (<>
    <div className="grid g-side">
      <Card>
        <CardHead icon="grid" title="Study design" sub="Every combination is generated and scored; per-cell seeds are derived deterministically" />
        <div className="col gap-16">
          <Field label="Name"><input className="input" value={name} onChange={e => setName(e.target.value)} /></Field>
          <div className="field">
            <label>Personas · {pids.length}</label>
            <div className="row wrap gap-6">
              {(personas.data?.personas || []).map(p => (
                <button key={p.id} className={`chip ${pids.includes(p.id) ? 'v' : ''}`} style={{ cursor: 'pointer' }} onClick={() => toggle(pids, setPids, p.id)}>
                  {pids.includes(p.id) && <Icon name="check" size={12} />}{p.label}
                </button>
              ))}
            </div>
          </div>
          <div className="field">
            <div className="row between"><label>Scenarios · {sids.length}</label>
              <select className="select" style={{ width: 170, height: 30, fontSize: 12 }} value={domain} onChange={e => setDomain(e.target.value)}>
                <option value="all">All domains</option>
                {Object.entries(DOMAIN_META).map(([k, m]) => <option key={k} value={k}>{m.label}</option>)}
              </select></div>
            <div className="col gap-4 scroll-y" style={{ maxHeight: 220 }}>
              {shown.map(s => (
                <label key={s.id} className="check" style={{ alignItems: 'flex-start', padding: '4px 0' }}>
                  <input type="checkbox" checked={sids.includes(s.id)} onChange={() => toggle(sids, setSids, s.id)} style={{ marginTop: 3 }} />
                  <span className="small">{s.text} <Badge tone={s.split === 'benchmark' ? 'cyan' : ''}>{s.split}</Badge></span>
                </label>
              ))}
            </div>
          </div>
          <Slider label="Repetitions" value={reps} min={1} max={10} onChange={setReps} />
          <Slider label="Temperature" value={temp} min={0.1} max={1.5} step={0.05} onChange={setTemp} format={v => v.toFixed(2)} />
          <Field label="Model"><select className="select" value={model} onChange={e => setModel(e.target.value)}><option value="active">Active model</option><option value="baseline">Baseline (frozen)</option></select></Field>
          <div className="row between">
            <span className="small muted"><span className="mono violet" style={{ fontSize: 18 }}>{pids.length}×{sids.length}×{reps}</span> = <b className="mono">{total}</b> trials</span>
            <Btn kind="primary" icon="play" disabled={pub || !total || running || total > 2000} onClick={start}>Run study</Btn>
          </div>
          {total > 2000 && <Note tone="warn">Maximum 2,000 trials per study.</Note>}
          {pub && <Note>Studies are disabled on the public site because they run hundreds of generations.</Note>}
          <ErrorBox error={err} />
        </div>
      </Card>
      <div className="col gap-16">
        {job && (
          <Card glow={running ? 'v' : undefined}>
            <CardHead icon="activity" title={running ? 'Running research study…' : job.status === 'completed' ? 'Study complete' : `Study ${job.status}`}
              right={<Badge tone={running ? 'violet' : job.status === 'completed' ? 'green' : 'red'} upper>{job.status}</Badge>} />
            <Progress value={job.progress} indeterminate={job.status === 'queued'} />
            <div className="row between small mt-8"><span className="muted">{job.message}</span><span className="mono">{Math.round((job.progress || 0) * 100)}%</span></div>
            {running && <StudyLive job={job} />}
            {job.status === 'failed' && <div className="mt-12"><ErrorBox>{job.error?.split('\n')[0]}</ErrorBox></div>}
          </Card>
        )}
        {openExp ? <StudyResult id={openExp} onClose={() => setOpenExp(null)} /> : (
          <Card>
            <CardHead icon="history" iconTone="cyan" title="Past studies" />
            {past.data?.experiments?.length ? past.data.experiments.map(e => (
              <button key={e.id} className="scen-item" style={{ marginBottom: 6 }} onClick={() => setOpenExp(e.id)}>
                <div className="row between"><b style={{ color: 'var(--text)' }}>{e.name}</b><Badge tone={fidelityTone(e.summary?.fidelity)}><span className="mono">{fmt(e.summary?.fidelity, 1)}</span></Badge></div>
                <div className="tiny dim mono mt-4">{e.id} · {e.model_version} · n={e.summary?.n ?? '–'} · {dateTime(e.created_at)} · {e.status}</div>
              </button>
            )) : <Empty icon="grid" title="No studies yet">Design a study on the left. Results appear here with a persona × scenario fidelity matrix.</Empty>}
          </Card>
        )}
      </div>
    </div>
  </>)
}

function StudyLive({ job }) {
  const recent = (job.events || []).filter(e => e.stage === 'evaluate' && e.data?.fidelity !== undefined).slice(-24)
  return (
    <div className="mt-12">
      <div className="section-label">Latest scores</div>
      <div className="row wrap gap-4">
        {recent.map((e, i) => <span key={i} className="pop-in" title={e.data.failures?.join(', ')} style={{ width: 26, height: 26, borderRadius: 7, display: 'grid', placeItems: 'center', fontSize: 9.5, fontFamily: 'var(--mono)', background: `${fidelityColor(e.data.fidelity)}26`, border: `1px solid ${fidelityColor(e.data.fidelity)}66`, color: fidelityColor(e.data.fidelity) }}>{Math.round(e.data.fidelity)}</span>)}
      </div>
    </div>
  )
}

function StudyResult({ id, onClose }) {
  const e = useApi(`/experiments/${id}`)
  const d = e.data
  const matrix = useMemo(() => {
    if (!d) return null
    const ps = [...new Map(d.trials.map(t => [t.persona_id, t.persona_snapshot.label])).entries()]
    const ss = [...new Map(d.trials.map(t => [t.scenario_snapshot.id || t.scenario_snapshot.text, t.scenario_snapshot])).entries()]
    const cell = {}
    d.trials.forEach(t => { const k = `${t.persona_id}|${t.scenario_snapshot.id || t.scenario_snapshot.text}`; (cell[k] = cell[k] || []).push(t.overall) })
    return { ps, ss, cell }
  }, [d])
  if (e.error) return <ErrorBox error={e.error} />
  if (!d) return <Card><Thinking>Loading study…</Thinking></Card>
  const s = d.summary || {}
  return (
    <Card className="pop-in">
      <CardHead icon="grid" title={d.name} sub={<span className="mono">{d.id} · {d.model_version} · {s.n} trials</span>}
        right={<button className="icon-btn" onClick={onClose} aria-label="Close"><Icon name="x" size={14} /></button>} />
      <div className="grid g4">
        {[['Fidelity', s.fidelity, 1, ''], ['Stability', s.stability, 1, ''], ['Consistency', s.consistency, 1, ''], ['Failure rate', s.failure_rate !== undefined ? s.failure_rate * 100 : null, 0, '%']].map(([k, v, dec, suf]) => (
          <div key={k} className="card elev" style={{ padding: 12 }}><div className="upper dim" style={{ fontSize: 10 }}>{k}</div>
            <div className="mono" style={{ fontSize: 22, marginTop: 4 }}>{v === null || v === undefined ? '–' : <CountUp value={v} decimals={dec} suffix={suf} />}</div></div>
        ))}
      </div>
      {s.stability === null && <p className="tiny dim">Stability and consistency need at least 2 repetitions per cell.</p>}
      <div className="section-label mt-16">Fidelity matrix (mean per cell)</div>
      <div className="table-wrap">
        <table className="t">
          <thead><tr><th>Persona</th>{matrix.ss.map(([k, sc]) => <th key={k} title={sc.text}><Icon name={DOMAIN_META[sc.domain]?.icon} size={13} /> {sc.text.slice(0, 22)}…</th>)}</tr></thead>
          <tbody>{matrix.ps.map(([pid, label]) => (
            <tr key={pid}><td className="nowrap small">{label}</td>
              {matrix.ss.map(([sk]) => {
                const v = matrix.cell[`${pid}|${sk}`]
                const m = v ? v.reduce((a, b) => a + b, 0) / v.length : null
                return <td key={sk} className="mono small" style={{ background: m === null ? undefined : `${fidelityColor(m)}1c`, color: fidelityColor(m) }}>{fmt(m, 1)}</td>
              })}</tr>
          ))}</tbody>
        </table>
      </div>
      {s.failure_rates && Object.keys(s.failure_rates).length > 0 && (<>
        <div className="section-label mt-16">Failure modes</div>
        <div className="row wrap">{Object.entries(s.failure_rates).map(([k, v]) => <Badge key={k} tone="red">{k.replace('_', ' ')} {pct(v)}</Badge>)}</div>
      </>)}
      {s.axes && (<>
        <div className="section-label mt-16">Mean axis scores</div>
        <div className="row wrap gap-16 small">{Object.entries(s.axes).map(([k, v]) => <span key={k}><span className="dim">{AXIS_LABEL[k]}</span> <b className="mono">{fmt(v, 2)}</b></span>)}</div>
      </>)}
    </Card>
  )
}

// ------------------------------------------------------------------ human evaluation
function HumanEval() {
  const { toast } = useApp()
  const [blind, setBlind] = useState(true)
  const [cur, setCur] = useState(null)
  const [loading, setLoading] = useState(false)
  const [score, setScore] = useState(null)
  const [corr, setCorr] = useState('')
  const [notes, setNotes] = useState('')
  const [err, setErr] = useState(null)
  const summary = useApi('/human-eval/summary')

  async function next() {
    setLoading(true); setErr(null); setScore(null); setCorr(''); setNotes('')
    try { setCur(await api(`/human-eval/next?blind=${blind}`)) } catch (e) { setErr(e) } finally { setLoading(false) }
  }
  useEffect(() => { next() }, [blind]) // eslint-disable-line react-hooks/exhaustive-deps
  async function submit() {
    try {
      const r = await api('/human-eval', { method: 'POST', body: { trial_id: cur.trial.id, score, blind, corrected_response: corr.trim() || null, notes } })
      toast(`Rating saved${r.examples_added.length ? ` · ${r.examples_added.join(', ')} added to the replay buffer` : ''}`, 'ok')
      summary.reload(true); next()
    } catch (e) { setErr(e) }
  }
  const t = cur?.trial
  const p = t?.persona_snapshot
  return (
    <div className="grid g-main">
      <Card>
        <CardHead icon={blind ? 'eyeOff' : 'eye'} title="Rate a twin answer" sub="Your 1–10 score is stored separately from the evaluator's 0–100 score and never mixed into it"
          right={<Toggle on={blind} onChange={setBlind} label="Blind (hide model)" />} />
        {loading && <Thinking>Fetching an unrated trial…</Thinking>}
        <ErrorBox error={err} />
        {!loading && cur && !t && <Empty icon="checkCircle" title="Nothing left to rate">Every stored trial already has a human rating. Run more experiments first.</Empty>}
        {!loading && t && (
          <div className="fade-in">
            <div className="card elev" style={{ padding: 14 }}>
              <div className="row between"><b>{p.label}</b>{!blind && <Badge tone="violet"><span className="mono">{t.model_version}</span></Badge>}</div>
              <div className="small muted">{[p.age && `${p.age} y/o`, p.gender, p.occupation, p.location].filter(Boolean).join(' · ')}</div>
              <div className="tiny dim mt-4">O {p.openness} · C {p.conscientiousness} · E {p.extraversion} · A {p.agreeableness} · N {p.neuroticism} · politics {p.political} · {p.commStyle}</div>
            </div>
            <div className="small muted mt-12">“{t.scenario_snapshot.text}”</div>
            <div className="response-box mt-8">{t.response || <span className="dim">(empty)</span>}</div>
            <div className="section-label mt-16">How well does this sound like the persona?</div>
            <div className="row wrap gap-6">
              {Array.from({ length: 10 }, (_, i) => i + 1).map(n => (
                <button key={n} onClick={() => setScore(n)} className="btn sm" style={{ width: 40, borderColor: score === n ? fidelityColor(n * 10) : undefined, background: score === n ? `${fidelityColor(n * 10)}22` : undefined, color: score === n ? fidelityColor(n * 10) : undefined }}>{n}</button>
              ))}
            </div>
            <div className="grid g2 mt-12">
              <Field label="Corrected response (optional)" hint="Becomes a human_correction training example (non-benchmark personas only)"><textarea className="textarea" rows={3} value={corr} onChange={e => setCorr(e.target.value)} /></Field>
              <Field label="Notes"><textarea className="textarea" rows={3} value={notes} onChange={e => setNotes(e.target.value)} /></Field>
            </div>
            <div className="row mt-12"><Btn kind="primary" icon="check" disabled={!score} onClick={submit}>Submit rating</Btn><Btn kind="ghost" onClick={next}>Skip</Btn>
              <span className="tiny dim">{cur.remaining} unrated trials</span></div>
          </div>
        )}
      </Card>
      <Card>
        <CardHead icon="scale" iconTone="cyan" title="Human vs evaluator" sub="Agreement between your ratings and the rubric" />
        {summary.data?.n ? (<>
          <div className="row gap-20">
            <div><div className="upper dim" style={{ fontSize: 10 }}>Ratings</div><div className="mono" style={{ fontSize: 24 }}>{summary.data.n}</div></div>
            <div><div className="upper dim" style={{ fontSize: 10 }}>Correlation</div><div className="mono" style={{ fontSize: 24 }}>{summary.data.human_vs_evaluator_correlation === null ? '–' : fmt(summary.data.human_vs_evaluator_correlation, 2)}</div></div>
          </div>
          {summary.data.human_vs_evaluator_correlation === null && <p className="tiny dim">Correlation needs at least 3 ratings with some variance.</p>}
          <table className="t mt-12"><thead><tr><th>Model</th><th className="r">n</th><th className="r">Human /10</th><th className="r">Evaluator /100</th></tr></thead>
            <tbody>{summary.data.by_model.map(r => <tr key={r.model_version}><td className="mono small">{r.model_version}</td><td className="r mono">{r.n}</td><td className="r mono">{fmt(r.human_mean, 2)}</td><td className="r mono">{fmt(r.evaluator_mean, 1)}</td></tr>)}</tbody></table>
          <p className="tiny dim mt-8">{summary.data.note}</p>
        </>) : <Empty icon="scale" title="No human ratings yet">Ratings measure whether the rubric agrees with people.</Empty>}
      </Card>
    </div>
  )
}
