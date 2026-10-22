import { useState } from 'react'
import Icon from '../icons.jsx'
import { AXIS_LABEL, DOMAIN_META, compact, fmt, fidelityTone, pct, signed, timeAgo, useApi } from '../api.js'
import { Bootstrap } from '../learning.jsx'
import { Badge, Btn, Card, CardHead, Empty, ErrorBox, Loading, Metric, useApp } from '../ui.jsx'
import { EvolutionTimeline, FidelityCurve, ModelDetail, PairedLabel, isAccepted } from '../components.jsx'

function HeroStatus({ data }) {
  const curve = data.curve || []
  const last = curve[curve.length - 1]
  const am = data.active_model
  return (
    <div className="hero-status">
      <div className="hs-row"><span className="dim">Active model</span><span className="mono">{am?.name}</span></div>
      <div className="hs-row"><span className="dim">Parameters</span><span className="mono">{am ? am.parameters.toLocaleString('en-US') : '–'}</span></div>
      <div className="hs-row"><span className="dim">Benchmark fidelity</span><span className="mono">{fmt(data.active_metrics?.fidelity, 2)} / 100</span></div>
      {last && last.model !== am?.name && (
        <div className="hs-row"><span className="dim">Latest candidate</span>
          <span className="mono">{last.model}: {fmt(last.fidelity, 2)}, {last.status}</span></div>
      )}
      <div className="hs-row"><span className="dim">Versions trained</span><span className="mono">{curve.length}</span></div>
    </div>
  )
}

const SOURCE_LABEL = { auto_best_of_n: 'best-of-N harvest', evaluator_feedback: 'trial feedback', human_rating: 'human ratings', human_correction: 'human corrections' }

export default function Dashboard() {
  const { nav, status } = useApp()
  const d = useApi('/metrics/dashboard', { poll: 20000 })
  const [sel, setSel] = useState(null)
  const data = d.data
  const untrained = status?.components?.llm?.state === 'untrained' || (data && !data.active_model)

  return (
    <div>
      <section className="hero">
        <div className="hero-copy">
          <h1>TwinSim</h1>
          <p className="hero-lead">
            A small language model, trained from scratch on this machine, answers questions as a described persona.
            A rule-based scorer checks how closely each answer matches the persona's personality and politics, and a fixed
            benchmark of 96 answers decides whether feedback training actually made the model better.
          </p>
          <div className="row wrap mt-16">
            <Btn kind="primary" icon="twin" onClick={() => nav('builder')}>Create a twin</Btn>
            <Btn icon="flask" onClick={() => nav('scenario')}>Run a scenario</Btn>
          </div>
        </div>
        {data && !untrained && <HeroStatus data={data} />}
      </section>

      <ErrorBox error={d.error} />
      {d.loading && !data && <Card><Loading rows={4} /></Card>}

      {data && untrained && (
        <Card glow="v">
          <Empty icon="brain" title="No model trained yet"
            >
            Every metric on this dashboard is measured from the baseline checkpoint. Create it below; from then on the model improves automatically as you run experiments.
          </Empty>
          <Bootstrap onDone={() => d.reload(true)} />
        </Card>
      )}

      {data && !untrained && <DashboardBody data={data} sel={sel} setSel={setSel} />}
    </div>
  )
}

function DashboardBody({ data, sel, setSel }) {
  const { nav } = useApp()
  const am = data.active_metrics
  const curve = data.curve || []
  const accepted = curve.filter(m => m.status !== 'baseline' && isAccepted(m)).length
  const rejected = curve.filter(m => m.status === 'rejected').length
  const total = Object.values(data.trial_counts || {}).reduce((a, b) => a + b, 0)
  const selected = curve.find(m => m.model === sel)
  const drift = Object.entries(data.drift || {})
  const bySource = Object.entries(data.training_examples_by_source || {})

  return (<>
    <div className="grid g4 stagger">
      <Metric i={0} icon="target" label="Active twin fidelity" value={am?.fidelity ?? null} decimals={1} suffix="/100"
        foot={data.improvement !== null && data.improvement !== undefined && data.active_model?.name !== data.baseline_model
          ? <><span className={data.improvement > 0 ? 'green' : 'red'}>{signed(data.improvement, 2)}</span> vs baseline · <PairedLabel p={data.paired_vs_baseline} short /></>
          : <>{data.active_model?.name} · fixed benchmark ({am?.details?.n ?? '–'} items)</>} />
      <Metric i={1} icon="flask" tone="cyan" label="Answers generated" value={total}
        foot={Object.entries(data.trial_counts || {}).map(([k, v]) => `${k} ${compact(v)}`).join(' · ') || 'none yet'} />
      <Metric i={2} icon="shield" label="Twin stability" value={am?.stability ?? null} decimals={1} suffix="/100"
        foot={`consistency ${fmt(am?.consistency, 1)} · ${accepted} of ${accepted + rejected} candidates accepted`} />
      <Metric i={3} icon="database" tone="cyan" label="Training examples" value={data.training_examples}
        foot={bySource.length ? bySource.map(([k, v]) => `${SOURCE_LABEL[k] || k.replace(/_/g, ' ')}: ${v}`).join(' · ') : 'replay buffer empty'} />
    </div>

    <div className="grid g-main mt-16">
      <Card i={4}>
        <CardHead icon="activity" title="Benchmark fidelity by version" sub="Every checkpoint answers the same 96 held-out prompts (6 personas × 8 scenarios × 2 seeds)"
          right={<div className="legend"><span><i style={{ background: '#E8772E' }} />accepted</span><span><i style={{ background: '#D96459' }} />rejected</span><span><i className="dashed" style={{ color: '#7FA7B8' }} />baseline</span></div>} />
        {curve.length ? <FidelityCurve curve={curve} /> : <Empty icon="activity" title="No benchmark yet" />}
        {curve.length === 1 && <p className="small dim" style={{ margin: '6px 0 0' }}>Only the baseline exists. Run experiments: the model learns from them in the background and new versions appear here once they measurably beat the active one.</p>}
      </Card>
      <Card>
        <CardHead icon="target" iconTone="cyan" title="Fidelity by axis" sub={`${data.active_model?.name} · mean axis score (0–10) on the benchmark`} />
        {am?.details?.axes ? (
          <div className="col gap-6">
            {Object.entries(am.details.axes).map(([k, v], i) => (
              <div key={k} className="tbar" style={{ '--i': i, margin: '3px 0' }}>
                <span className="muted">{AXIS_LABEL[k]}</span>
                <div className="track"><div className="fill" style={{ width: `${v * 10}%`, background: '#E8772E' }} /></div>
                <span className="mono small" style={{ textAlign: 'right' }}>{fmt(v, 1)}</span>
              </div>
            ))}
            <div className="divider" />
            <div className="row between small"><span className="muted">Stability</span><span className="mono">{fmt(am.stability, 1)}</span></div>
            <div className="row between small"><span className="muted">Consistency</span><span className="mono">{fmt(am.consistency, 1)}</span></div>
            <div className="row between small"><span className="muted">Failure rate</span><span className="mono">{pct(am.failure_rate, 1)}</span></div>
          </div>
        ) : <Empty icon="target" title="No axis data" />}
      </Card>
    </div>

    <Card className="mt-16">
      <CardHead icon="route" title="Model evolution" sub="Every checkpoint TwinSim has produced. Select a node for its acceptance decision."
        right={<Btn size="sm" kind="ghost" icon="layers" onClick={() => nav('registry')}>Registry</Btn>} />
      <EvolutionTimeline curve={curve} selected={sel} onSelect={setSel} />
      {selected && <ModelDetail m={selected} onClose={() => setSel(null)} />}
    </Card>

    <div className="grid g2 mt-16">
      <Card>
        <CardHead icon="compare" title="Persona drift" sub={`Mean |observed − intended| per benchmark persona · ${data.active_model?.name}`}
          right={<Btn size="sm" kind="ghost" icon="dna" onClick={() => nav('dna')}>Twin DNA</Btn>} />
        {drift.length ? drift.map(([pid, d], i) => {
          const vals = Object.values(d.drift || {})
          const mad = vals.length ? vals.reduce((a, b) => a + Math.abs(b), 0) / vals.length : 0
          const col = mad < 1 ? '#5FB37A' : mad < 2.5 ? '#D9A441' : '#D96459'
          return (
            <div key={pid} className="tbar" style={{ '--i': i }}>
              <span className="muted ellipsis" title={d.label}>{d.label}</span>
              <div className="track"><div className="fill" style={{ width: `${Math.min(10, mad) * 10}%`, background: col }} /></div>
              <span className="mono small" style={{ textAlign: 'right', color: col }}>{fmt(mad, 2)}</span>
            </div>
          )
        }) : <Empty icon="compare" title="No drift data" />}
      </Card>
      <Card>
        <CardHead icon="history" iconTone="cyan" title="Recent experiments" sub="Latest twin answers (user, study and demo trials)"
          right={<Btn size="sm" kind="ghost" icon="flask" onClick={() => nav('scenario')}>New</Btn>} />
        {data.recent?.length ? (
          <div className="col gap-4">
            {data.recent.map(r => {
              const dm = DOMAIN_META[r.domain] || {}
              return (
                <div key={r.id} className="row" style={{ padding: '8px 0', borderBottom: '1px solid rgba(46,41,36,.5)' }}>
                  <span style={{ color: dm.color }}><Icon name={dm.icon || 'chat'} size={16} /></span>
                  <div className="grow"><div className="small ellipsis">{r.scenario}</div>
                    <div className="tiny dim">{r.persona} · <span className="mono">{r.model}</span> · {timeAgo(r.created_at)}</div></div>
                  {r.failures.length > 0 && <Badge tone="amber">{r.failures.length} flag{r.failures.length > 1 ? 's' : ''}</Badge>}
                  <Badge tone={fidelityTone(r.fidelity)}><span className="mono">{fmt(r.fidelity, 1)}</span></Badge>
                </div>
              )
            })}
          </div>
        ) : (
          <Empty icon="flask" title="No experiments yet" action={<Btn kind="primary" icon="twin" onClick={() => nav('builder')}>Create Twin</Btn>}>
            Create your first digital twin to begin research.
          </Empty>
        )}
      </Card>
    </div>
  </>)
}
