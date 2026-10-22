import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import Icon from '../icons.jsx'
import { AXES, AXIS_LABEL, DOMAIN_META, TRAITS, api, fmt, streamTrial, useApi } from '../api.js'
import { DriftPlot, TraitBars } from '../charts.jsx'
import { Badge, Btn, Card, CardHead, Empty, ErrorBox, Field, Gauge, Modal, Note, PageHead, Seg, Slider, Thinking, Toggle, copyText, useApp } from '../ui.jsx'

// Frontend state machine; each transition is driven by a backend event (SSE stage).
// idle -> preparing -> generating -> evaluating -> feedback -> completed | failed
// The last pipeline node (Auto-learning) is automatic: it only reports what the backend queued, nothing is started here.
const PIPE = [
  { id: 'persona', label: 'Persona', icon: 'user' },
  { id: 'generate', label: 'Transformer', icon: 'network' },
  { id: 'evaluate', label: 'Evaluator', icon: 'gauge' },
  { id: 'feedback', label: 'Failure analysis', icon: 'database' },
  { id: 'learn', label: 'Auto-learning', icon: 'brain' },
]
const PHASE_IDX = { preparing: 0, generating: 1, evaluating: 2, feedback: 3 }
function stageStates(phase, failedAt) {
  const n = PIPE.length
  if (phase === 'idle') return Array(n).fill('')
  if (phase === 'completed') return Array(n).fill('done')
  if (phase === 'failed') return PIPE.map((_, i) => (i < failedAt ? 'done' : i === failedAt ? 'failed' : ''))
  const cur = PHASE_IDX[phase]
  return PIPE.map((_, i) => (i < cur ? 'done' : i === cur ? 'active' : ''))
}
const TRAINING_STATUS = {
  pending: ['green', 'kept as a positive example for the next learning update'],
  eligible: ['green', 'eligible for training'],
  not_eligible: ['amber', 'below the fidelity gate: analysed for corrections, not used as a positive example'],
  excluded_benchmark: ['cyan', 'benchmark persona or scenario: held out of learning so evaluation stays honest'],
  not_recorded: ['', 'learning disabled for this run'],
}
const FAILURE_INFO = {
  degeneration: ['Output could not be measured', 'The answer is too short, unreadable or stuck in a loop, so the evaluator sets fidelity to 0.', 'All axes'],
  exaggeration: ['Trait was over-expressed', 'A trait was pushed further from neutral than the persona specifies (2.5+ points on the 0–10 scale), or the answer leaned on many intensifiers.', 'Personality'],
  blue_shift: ['Politics drifted progressive', 'The expressed political position was 2.5+ points more progressive than the persona.', 'Politics'],
  red_shift: ['Politics drifted conservative', 'The expressed political position was 2.5+ points more conservative than the persona.', 'Politics'],
  stereotype: ['Demographic stereotype', "The answer matched a demographic archetype for this persona's age or occupation rather than the persona's own trait values.", 'Personality / politics'],
  inconsistency: ['Contradictory signals', 'The answer contains opposing cues, e.g. both supporting and opposing the proposal, or mixed trait evidence.', 'Consistency'],
}
const LABEL = { exaggeration: 'Generative Exaggeration', blue_shift: 'Blue-Shift Bias', red_shift: 'Red-Shift Bias', stereotype: 'Stereotype Projection', inconsistency: 'Temporal Inconsistency', degeneration: 'Degenerate Output' }

function Pipeline({ phase, info }) {
  const states = stageStates(phase, info.failedAt ?? 0)
  return (
    <div className="pipeline">
      {PIPE.map((p, i) => {
        const state = states[i], next = states[i + 1]
        return (
          <div key={p.id} className={`pstage ${state}`}>
            {i < PIPE.length - 1 && <div className={`plink ${next === 'done' || next === 'active' || next === 'failed' ? 'done' : ''} ${next === 'active' ? 'flowing' : ''}`}><i /></div>}
            <div className="pnode"><Icon name={state === 'failed' ? 'xCircle' : state === 'done' && i === PIPE.length - 1 ? 'checkCircle' : p.icon} size={20} /></div>
            <div className="pname">{p.label}</div>
            <div className="pdetail" title={info[p.id]}>{info[p.id] || '\u00a0'}</div>
          </div>
        )
      })}
    </div>
  )
}

// Decorative activity indicator: lights only while the backend reports the generating stage.
function NeuralActivity({ active }) {
  const cols = [4, 6, 6, 4]
  const nodes = cols.flatMap((n, c) => Array.from({ length: n }, (_, k) => ({ x: 14 + c * 38, y: 50 + (k - (n - 1) / 2) * 14, c })))
  return (
    <svg width="140" height="100" viewBox="0 0 140 100" aria-hidden="true">
      {nodes.map((a, i) => nodes.filter(b => b.c === a.c + 1).map((b, j) => <line key={`${i}-${j}`} className="e" x1={a.x} y1={a.y} x2={b.x} y2={b.y} />))}
      {nodes.map((n, i) => <circle key={i} className={`n ${active ? 'on' : ''}`} cx={n.x} cy={n.y} r="3" style={{ animationDelay: `${(i * 137) % 900}ms`, opacity: active ? undefined : 0.3 }} />)}
    </svg>
  )
}

function download(name, obj) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' }))
  const a = Object.assign(document.createElement('a'), { href: url, download: name })
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export default function ScenarioLab({ params }) {
  const { toast, twinDraft } = useApp()
  const personas = useApi('/personas')
  const scen = useApi('/scenarios')
  const models = useApi('/models')
  const [domain, setDomain] = useState(params?.domain || 'political')
  const [personaId, setPersonaId] = useState(params?.persona || 'p1')
  const [scenarioId, setScenarioId] = useState(null)
  const [custom, setCustom] = useState('')
  const [mode, setMode] = useState('library')
  const [model, setModel] = useState('active')
  const [gen, setGen] = useState({ temperature: 0.8, top_k: 40, top_p: 0.95, repetition_penalty: 1.15, max_new_tokens: 96, seed: '', deterministic: false })
  const [learn, setLearn] = useState(true)
  const [phase, setPhase] = useState('idle')
  const [info, setInfo] = useState({})
  const [text, setText] = useState('')
  const [tokens, setTokens] = useState(0)
  const [result, setResult] = useState(null)
  const [err, setErr] = useState(null)
  const [extra, setExtra] = useState(null)
  const [cmpOpen, setCmpOpen] = useState(false)
  const abort = useRef(null)
  const outRef = useRef(null)

  const domainScen = useMemo(() => (scen.data?.scenarios || []).filter(s => s.domain === domain), [scen.data, domain])
  useEffect(() => { if (domainScen.length && !domainScen.find(s => s.id === scenarioId)) setScenarioId((domainScen.find(s => s.split !== 'benchmark') || domainScen[0]).id) }, [domainScen]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => abort.current?.abort(), [])

  const personaList = personas.data?.personas || []
  const draft = personaId === 'draft' && twinDraft
  const persona = draft ? twinDraft : personaList.find(p => p.id === personaId)
  const scenario = mode === 'custom' ? { domain, text: custom } : domainScen.find(s => s.id === scenarioId)
  const busy = ['preparing', 'generating', 'evaluating', 'feedback'].includes(phase)
  const canRun = persona && scenario && (mode !== 'custom' || custom.trim().length >= 10) && !busy
  const usable = (models.data?.models || []).filter(m => !m.weights_pruned)

  async function run() {
    setErr(null); setResult(null); setText(''); setTokens(0); setExtra(null)
    setPhase('preparing'); setInfo({ persona: persona?.label || 'custom twin' })
    const body = {
      ...(draft ? { persona: twinDraft } : { persona_id: personaId }),
      ...(mode === 'custom' ? { scenario: { domain, text: custom.trim() } } : { scenario_id: scenarioId }),
      model_version: model, learn,
      generation: { ...gen, seed: gen.seed === '' ? null : Number(gen.seed), temperature: gen.deterministic ? 0 : gen.temperature },
    }
    abort.current = new AbortController()
    try {
      await streamTrial(body, ev => {
        if (ev.stage === 'preparing') { if (ev.model) setInfo(i => ({ ...i, persona: ev.persona || i.persona, generate: ev.model })) }
        else if (ev.stage === 'generating') {
          setPhase('generating')
          if (ev.text !== undefined) { setText(ev.text); setTokens(ev.tokens); setInfo(i => ({ ...i, generate: `${ev.tokens} tokens` })) }
          else if (ev.model) setInfo(i => ({ ...i, generate: ev.model }))
        } else if (ev.stage === 'evaluating') { setPhase('evaluating'); setInfo(i => ({ ...i, generate: `${ev.tokens} tok · ${Math.round(ev.latency_ms)} ms`, evaluate: 'rubric scoring' })) }
        else if (ev.stage === 'feedback') { setPhase('feedback'); setInfo(i => ({ ...i, evaluate: `fidelity ${fmt(ev.overall, 1)}`, feedback: 'finding failure modes' })) }
        else if (ev.stage === 'completed') {
          const r = ev.result, L = r.learning || {}
          setResult(r); setText(r.trial.response); setPhase('completed')
          const nFail = r.evaluation.failures.length
          const learnText = r.trial.training_status === 'excluded_benchmark' ? 'held out'
            : L.scheduled ? 'update started' : L.reason ? 'off' : L.pending_signals !== undefined ? `${L.pending_signals}/${L.threshold} queued` : 'queued'
          setInfo(i => ({ ...i, feedback: nFail ? `${nFail} failure mode${nFail > 1 ? 's' : ''}` : 'no failures', learn: learnText }))
          if (window.innerWidth < 1280) setTimeout(() => outRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120)
        } else if (ev.stage === 'failed') {
          throw Object.assign(new Error(ev.error?.message || 'trial failed'), { code: ev.error?.code })
        }
      }, abort.current.signal)
    } catch (e) {
      if (e.name === 'AbortError') return
      setErr(e); setPhase('failed')
      setInfo(i => ({ ...i, failedAt: i.evaluate ? 2 : i.generate ? 1 : 0 }))
    }
  }

  async function replay() {
    try {
      const r = await api(`/trials/${result.trial.id}/replay`, { method: 'POST' })
      setExtra({ kind: 'replay', ...r })
      toast(r.identical ? 'Replay produced the identical response' : 'Replay output differs', r.identical ? 'ok' : 'err')
    } catch (e) { setErr(e) }
  }
  async function reevaluate() {
    try {
      const r = await api('/evaluate', { method: 'POST', body: { persona: result.trial.persona_snapshot, scenario: result.trial.scenario_snapshot.text, response: result.trial.response } })
      setExtra({ kind: 'eval', overall: r.overall, same: Math.abs(r.overall - result.evaluation.overall) < 1e-6 })
    } catch (e) { setErr(e) }
  }

  const ev = result?.evaluation
  const L = result?.learning

  return (<>
    <PageHead eyebrow={<><Icon name="flask" size={13} />Experiment</>} title="Scenario Lab"
      sub="Pick a twin and a scenario and run it. The answer streams in as the model writes it, the rubric scorer rates it, and the system learns from the result in the background." />

    <div className="lab-domains" role="tablist" aria-label="Scenario category">
      {Object.entries(DOMAIN_META).map(([k, m]) => {
        const n = (scen.data?.scenarios || []).filter(s => s.domain === k).length
        return (
          <button key={k} role="tab" aria-selected={domain === k} className={`lab-domain ${domain === k ? 'on' : ''}`} onClick={() => { setDomain(k); setMode('library') }}>
            <span className="lab-domain-ic" style={{ color: m.color }}><Icon name={m.icon} size={18} /></span>
            <span className="grow"><b>{m.label}</b></span>
            <span className="tiny dim mono">{scen.data ? n : '…'}</span>
          </button>
        )
      })}
    </div>

    <div className="lab-grid mt-12">
      <Card className="lab-setup">
        <CardHead icon="settings" title="Experiment setup" sub="Scenario, twin and generation settings. Every finished run is analysed and used for learning automatically."
          right={<Badge>{DOMAIN_META[domain]?.label}</Badge>} />
        <div className="lab-setup-cols">
          <section className="lab-col">
            <div className="row between"><label className="label">Scenario</label><Seg value={mode} onChange={setMode} options={[['library', 'Library'], ['custom', 'Write your own']]} /></div>
            {mode === 'library' ? (
              <div className="scen-list">
                {domainScen.map(s => (
                  <button key={s.id} className={`scen-item ${scenarioId === s.id ? 'on' : ''}`} onClick={() => setScenarioId(s.id)}>
                    {s.text}
                    <div className="row gap-6 mt-4"><Badge tone={s.split === 'benchmark' ? 'cyan' : s.split === 'train' ? 'violet' : ''}>{s.split}</Badge>{s.issue && <span className="tiny dim mono">{s.issue}</span>}</div>
                  </button>
                ))}
              </div>
            ) : (
              <textarea className="textarea scen-custom" rows={12} value={custom} onChange={e => setCustom(e.target.value)} placeholder={`A ${DOMAIN_META[domain].label.toLowerCase()} situation, 10–600 characters`} />
            )}
          </section>

          <section className="lab-col">
            <Field label="Twin">
              <select className="select" value={personaId} onChange={e => setPersonaId(e.target.value)}>
                {twinDraft && <option value="draft">Unsaved twin from builder: {twinDraft.label || 'draft'}</option>}
                {personaList.map(p => <option key={p.id} value={p.id}>{p.label}{p.is_benchmark ? ' (benchmark)' : ''}</option>)}
              </select>
            </Field>
            {persona ? <PersonaSummary persona={persona} /> : <Note tone="warn">{personaId === 'draft' ? 'The unsaved twin is gone (page reloaded). Pick a saved twin or rebuild it.' : 'Loading twin…'}</Note>}
            <Field label="Model">
              <select className="select" value={model} onChange={e => setModel(e.target.value)}>
                <option value="active">Active model{models.data?.active ? ` (${models.data.active})` : ''}</option>
                <option value="baseline">Baseline (TwinSim-v0, frozen)</option>
                {usable.map(m => <option key={m.name} value={m.name}>{m.name} · {m.status}</option>)}
              </select>
            </Field>
            <div className="lab-gen">
              <Slider label="Temperature" value={gen.temperature} min={0.1} max={1.5} step={0.05} onChange={v => setGen(g => ({ ...g, temperature: v }))} format={v => v.toFixed(2)} />
              <Slider label="Top-k" value={gen.top_k} min={0} max={200} onChange={v => setGen(g => ({ ...g, top_k: v }))} />
              <Slider label="Top-p" value={gen.top_p} min={0.1} max={1} step={0.01} onChange={v => setGen(g => ({ ...g, top_p: v }))} format={v => v.toFixed(2)} />
              <Slider label="Repetition" value={gen.repetition_penalty} min={1} max={2} step={0.01} onChange={v => setGen(g => ({ ...g, repetition_penalty: v }))} format={v => v.toFixed(2)} />
              <Slider label="Max tokens" value={gen.max_new_tokens} min={16} max={256} step={8} onChange={v => setGen(g => ({ ...g, max_new_tokens: v }))} />
            </div>
            <div className="grid g2">
              <Field label="Seed" hint="blank = random (recorded)"><input className="input mono" value={gen.seed} onChange={e => setGen(g => ({ ...g, seed: e.target.value.replace(/[^0-9]/g, '') }))} placeholder="random" /></Field>
              <div className="col gap-6" style={{ justifyContent: 'flex-end' }}>
                <Toggle on={gen.deterministic} onChange={v => setGen(g => ({ ...g, deterministic: v }))} label="Greedy decoding" />
                <Toggle on={learn} onChange={setLearn} label="Let the model learn from this run" />
              </div>
            </div>
          </section>
        </div>
        <div className="lab-run">
          <Btn kind="primary" size="lg" icon={busy ? null : 'play'} loading={busy} disabled={!canRun} onClick={run}>
            {phase === 'generating' ? 'Twin is thinking…' : phase === 'evaluating' ? 'Measuring fidelity…' : busy ? 'Analysing…' : 'Run experiment'}
          </Btn>
        </div>
      </Card>

      <div className="lab-out col gap-16" ref={outRef}>
        <Card glow={busy ? 'v' : undefined}>
          <CardHead icon="route" title="Research pipeline" sub="Each stage lights up only when the backend reports it"
            right={<Badge tone={phase === 'failed' ? 'red' : phase === 'completed' ? 'green' : busy ? 'violet' : ''} upper>{phase}</Badge>} />
          <Pipeline phase={phase} info={info} />
        </Card>

        {phase === 'idle' && !result && (
          <Card><Empty icon="flask" title="No experiment running">Pick a twin and a scenario, then run the experiment. Nothing is shown until the model actually generates it.</Empty></Card>
        )}

        {(busy || (phase === 'failed' && !result && text)) && (
          <div className="genviz fade-in">
            <div className="row between">
              <div className="row"><NeuralActivity active={phase === 'generating'} />
                <div>
                  <div style={{ fontWeight: 600 }}>{phase === 'preparing' ? 'Loading checkpoint & serialising persona…' : phase === 'generating' ? 'Twin is thinking…' : 'Measuring behavioral fidelity…'}</div>
                  <div className="token-count mt-4"><Icon name="bolt" size={12} />{tokens} tokens</div>
                </div>
              </div>
              <span className="tiny dim mono">{info.generate}</span>
            </div>
            <div className="divider" />
            <div className="stream-text">{text}{phase === 'generating' && <span className="caret" />}</div>
            {phase === 'evaluating' && <Thinking>Rubric evaluator scoring 7 axes…</Thinking>}
          </div>
        )}
        <ErrorBox error={err} />

        {result && ev && (<>
          <Card>
            <CardHead icon="chat" title="Twin response" sub={`${persona?.label || result.trial.persona_snapshot.label} · ${DOMAIN_META[result.trial.scenario_snapshot.domain]?.label}`}
              right={<Badge tone="violet"><span className="mono">{result.trial.model_version}</span></Badge>} />
            <div className="small muted" style={{ marginBottom: 10 }}>“{result.trial.scenario_snapshot.text}”</div>
            <div className="response-box">{result.trial.response || <span className="dim">(empty response)</span>}</div>
            <div className="meta-row mt-12">
              <span>temperature <b>{fmt(result.trial.gen_params.temperature, 2)}</b></span>
              <span>seed <b>{result.trial.seed}</b></span>
              <span>top-k/p <b>{result.trial.gen_params.top_k}/{result.trial.gen_params.top_p}</b></span>
              <span>tokens <b>{result.generation.tokens_generated}</b></span>
              <span>latency <b>{Math.round(result.generation.latency_ms)} ms</b></span>
              <span>stop <b>{result.generation.stop_reason}</b></span>
              <span>backend <b>{result.trial.backend}</b></span>
            </div>
            <div className="row wrap mt-16">
              <Btn size="sm" icon="copy" onClick={() => copyText(result.trial.response, toast)}>Copy</Btn>
              <Btn size="sm" icon="replay" onClick={replay}>Replay</Btn>
              <Btn size="sm" icon="gauge" onClick={reevaluate}>Re-evaluate</Btn>
              <Btn size="sm" icon="compare" onClick={() => setCmpOpen(true)}>Compare</Btn>
              <Btn size="sm" icon="download" onClick={() => download(`${result.trial.id}.json`, result)}>Save JSON</Btn>
            </div>
            {extra?.kind === 'replay' && (
              <div className="mt-12"><Note tone={extra.identical ? 'info' : 'warn'} icon={extra.identical ? 'checkCircle' : 'warn'}>
                Replay of <span className="mono">{result.trial.id}</span> with the stored seed and parameters: <b>{extra.identical ? 'identical output' : 'different output'}</b>
                {!extra.identical && <div className="mt-4 small">{extra.replay.trial.response}</div>}
              </Note></div>
            )}
            {extra?.kind === 'eval' && <div className="mt-12"><Note>Evaluator re-run: <b className="mono">{fmt(extra.overall, 2)}</b> {extra.same ? '(identical, the rubric is deterministic)' : '(differs: evaluator settings changed since)'}</Note></div>}
          </Card>
          <Card>
            <div className="row gap-16 wrap" style={{ alignItems: 'center' }}>
              <Gauge value={ev.overall} size={150} sub={ev.valid ? `${ev.evaluator}` : 'invalid output'} />
              <div className="grow" style={{ minWidth: 200 }}>
                <div className="row wrap">
                  {ev.failures.length === 0 ? <Badge tone="green">no failure modes</Badge> : ev.failures.map((f, i) => <Badge key={i} tone="red">{LABEL[f.code] || f.code}</Badge>)}
                </div>
                <div className="tiny dim mt-8">Rubric score, not a human judgement.</div>
                <div className="small muted mt-8">{TRAINING_STATUS[result.trial.training_status]?.[1]}
                  {L?.pending_signals !== undefined && result.trial.training_status !== 'excluded_benchmark' && !L.scheduled && <> · {L.pending_signals}/{L.threshold} experiments queued for the next learning update</>}
                  {L?.scheduled && <> · learning update started in the background</>}</div>
              </div>
            </div>
          </Card>
        </>)}
      </div>
    </div>

    {result && ev && (
      <div className="grid g3 mt-16 stagger">
        <Card style={{ '--i': 0 }}>
          <CardHead icon="target" iconTone="cyan" title="Fidelity by axis" sub="10 − |intended − observed|, per axis (0–10)" />
          <TraitBars rows={AXES.map(a => ({ key: a, label: AXIS_LABEL[a], value: ev.axes[a] ?? 0 }))} />
        </Card>
        <Card style={{ '--i': 1 }}>
          <CardHead icon="compare" title="Persona drift" sub="○ intended → ● observed (0–10)" />
          {ev.valid ? <DriftPlot axes={[...TRAITS, 'political']} intended={ev.intended} observed={ev.observed} /> : <Empty icon="warn" title="Not measurable">Degenerate output has no observable traits.</Empty>}
        </Card>
        <div style={{ '--i': 2 }}><Diagnostic ev={ev} /></div>
      </div>
    )}
    <CompareModal open={cmpOpen} onClose={() => setCmpOpen(false)} result={result} models={(models.data?.models || []).filter(m => !m.weights_pruned)} />
  </>)
}

// Who the twin is: the persona sliders (0-10 here) and the descriptive fields the model is conditioned on.
function PersonaSummary({ persona }) {
  const rows = [...TRAITS.map(t => ({ key: t, label: AXIS_LABEL[t], value: (persona[t] ?? 50) / 10, color: 'var(--measure)' })),
    { key: 'political', label: 'Political lean', value: (persona.political ?? 50) / 10, color: 'var(--accent)' }]
  const facts = [['Age', persona.age], ['Gender', persona.gender], ['Occupation', persona.occupation], ['Education', persona.education],
    ['Location', persona.location], ['Economy', persona.economic], ['Style', persona.commStyle]].filter(([, v]) => v)
  return (
    <div className="persona-sum">
      <TraitBars rows={rows} />
      <div className="tiny dim" style={{ margin: '2px 0 8px' }}>0 = low / left · 10 = high / right</div>
      <dl className="kv" style={{ gridTemplateColumns: '84px minmax(0, 1fr)' }}>
        {facts.map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd>{v}</dd></Fragment>)}
      </dl>
    </div>
  )
}

function Diagnostic({ ev }) {
  const fails = ev.failures || []
  return (
    <div className={`diag ${fails.length ? '' : 'ok'}`}>
      <div className="row between"><span className={`upper ${fails.length ? 'red' : 'green'}`}>{fails.length ? 'What went wrong?' : 'No failure detected'}</span>
        <Icon name={fails.length ? 'alert' : 'shield'} size={16} className={fails.length ? 'red' : 'green'} /></div>
      {fails.length === 0 && <p className="small muted">None of the six failure detectors fired. Remaining error is spread across the axes on the left.</p>}
      {fails.map((f, i) => {
        const [title, expl, scope] = FAILURE_INFO[f.code] || [f.code, '', '']
        const trait = (f.detail.match(/^([a-z]+):/) || [])[1]
        const drift = trait && ev.drift?.[trait]
        return (
          <div key={i} className="diag-item">
            <Icon name="xCircle" size={18} className="red" />
            <div>
              <div style={{ fontWeight: 600 }}>{LABEL[f.code] || f.code}</div>
              <div className="small muted">{title}. {expl}</div>
              <dl className="kv">
                <dt>Evidence</dt><dd className="mono" style={{ fontSize: 11.5 }}>{f.detail}</dd>
                <dt>Trait affected</dt><dd>{trait ? AXIS_LABEL[trait] || trait : scope}</dd>
                {trait && ev.axes?.[trait] !== undefined && <><dt>Impact</dt><dd>{AXIS_LABEL[trait]} axis scored <b className="mono">{fmt(ev.axes[trait], 1)}</b>/10</dd></>}
                {f.code === 'degeneration' && <><dt>Impact</dt><dd>Fidelity forced to 0</dd></>}
                {drift !== undefined && drift !== null && <><dt>Drift</dt><dd className="mono">{drift >= 0 ? '+' : ''}{fmt(drift, 2)}</dd></>}
              </dl>
            </div>
          </div>
        )
      })}
      {ev.explanation?.length > 0 && (
        <div className="mt-8">
          <div className="section-label">Evaluator notes</div>
          <ul className="list-plain small muted">{ev.explanation.slice(0, 5).map((x, i) => <li key={i}>{x}</li>)}</ul>
        </div>
      )}
    </div>
  )
}

function CompareModal({ open, onClose, result, models }) {
  const [other, setOther] = useState('baseline')
  const [state, setState] = useState(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => { if (!open) setState(null) }, [open])
  if (!result) return null
  async function go() {
    setBusy(true); setState(null)
    try {
      const t = result.trial
      const r = await api('/experiments/trial', { method: 'POST', body: {
        persona: t.persona_snapshot, scenario_id: t.scenario_snapshot.id || undefined,
        scenario: t.scenario_snapshot.id ? undefined : { domain: t.scenario_snapshot.domain, text: t.scenario_snapshot.text },
        model_version: other, learn: false, generation: t.gen_params } })
      setState(r)
    } catch (e) { setState({ error: e }) } finally { setBusy(false) }
  }
  const a = result
  return (
    <Modal open={open} onClose={onClose} width={860}>
      <div className="row between"><h3 className="card-title"><Icon name="compare" size={16} />Same twin, same scenario, same seed: different model</h3>
        <button className="icon-btn" onClick={onClose} aria-label="Close"><Icon name="x" size={14} /></button></div>
      <div className="row mt-12">
        <select className="select" style={{ maxWidth: 320 }} value={other} onChange={e => setOther(e.target.value)}>
          <option value="baseline">Baseline (TwinSim-v0)</option>
          {models.map(m => <option key={m.name} value={m.name}>{m.name} · {m.status}</option>)}
        </select>
        <Btn kind="primary" icon="play" loading={busy} onClick={go}>Run comparison</Btn>
      </div>
      {busy && <Thinking>Generating with the other checkpoint…</Thinking>}
      {state?.error && <div className="mt-12"><ErrorBox error={state.error} /></div>}
      {state?.trial && (
        <div className="grid g2 mt-16">
          {[a, state].map((r, i) => (
            <div key={i} className="card elev">
              <div className="row between"><Badge tone={i ? 'cyan' : 'violet'}><span className="mono">{r.trial.model_version}</span></Badge>
                <span className="mono" style={{ fontSize: 22, fontWeight: 700 }}>{fmt(r.evaluation.overall, 1)}</span></div>
              <div className="response-box mt-12" style={{ fontSize: 13.5 }}>{r.trial.response}</div>
              <div className="row wrap mt-8">{r.evaluation.failures.map((f, k) => <Badge key={k} tone="red">{f.code}</Badge>)}</div>
            </div>
          ))}
        </div>
      )}
      {state?.trial && <p className="tiny dim mt-12">Stored as an ordinary trial with learning disabled. A single pair is anecdotal; the benchmark comparison in the Model Registry uses 96 paired items.</p>}
    </Modal>
  )
}
