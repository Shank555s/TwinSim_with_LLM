import { useState } from 'react'
import Icon from '../icons.jsx'
import { pct, useApi } from '../api.js'
import { LineChart } from '../charts.jsx'
import { Badge, Card, CardHead, Empty, ErrorBox, Loading, PageHead, Seg, useApp } from '../ui.jsx'

const COLORS = { degeneration: '#D96459', exaggeration: '#D9A441', blue_shift: '#6F95C9', red_shift: '#C47A9A', stereotype: '#D99A6C', inconsistency: '#7FA7B8' }
const DESC = {
  degeneration: 'Too short, unreadable or looping; fidelity is forced to 0.',
  exaggeration: 'A trait pushed 2.5+ points beyond the persona, or heavy intensifiers.',
  blue_shift: 'Expressed politics 2.5+ points more progressive than intended.',
  red_shift: 'Expressed politics 2.5+ points more conservative than intended.',
  stereotype: "Matched a demographic archetype instead of the persona's own traits.",
  inconsistency: 'Contradictory stance or trait signals within one answer.',
}

export default function Failures() {
  const { nav } = useApp()
  const f = useApi('/metrics/failures')
  const curve = useApi('/metrics/learning-curve')
  const [scope, setScope] = useState('benchmark')
  const d = f.data
  if (f.error) return <ErrorBox error={f.error} />
  if (!d) return <Card><Loading rows={5} /></Card>
  const codes = Object.keys(d.labels)
  const rates = scope === 'benchmark' ? d.current || {} : d.all_trials?.rates || {}
  const max = Math.max(0.0001, ...codes.map(c => rates[c] || 0))
  const versions = (curve.data?.curve || [])
  const vIdx = Object.fromEntries(versions.map(m => [m.model, m.version]))
  const anyTrend = codes.some(c => (d.trend[c] || []).length)
  const anyFail = codes.some(c => rates[c])

  return (<>
    <PageHead eyebrow={<><Icon name="alert" size={13} />Intelligence</>} title="Failure Analysis"
      sub="Six rule-based detectors run on every answer. Rates on the benchmark are comparable across model versions; rates over all trials depend on what was run."
      right={<Seg value={scope} onChange={setScope} options={[['benchmark', `Benchmark · ${d.active_model || '–'}`], ['all', `All trials · n=${d.all_trials?.n ?? 0}`]]} />} />
    {!d.active_model && <Card><Empty icon="alert" title="No model trained yet">Failure statistics appear after the first benchmark.</Empty></Card>}
    {d.active_model && (<>
      <div className="grid g-main">
        <Card>
          <CardHead icon="alert" iconTone="red" title="Failure distribution" sub={scope === 'benchmark' ? `Share of benchmark answers of ${d.active_model} flagged by each detector` : 'Share of all evaluated user/study/demo trials'} />
          {anyFail ? codes.map((c, i) => (
            <div key={c} className="tbar" style={{ '--i': i, gridTemplateColumns: '190px minmax(0,1fr) 58px', margin: '14px 0' }}>
              <span><span style={{ color: COLORS[c] }}>●</span> {d.labels[c]}</span>
              <div className="track" style={{ height: 10 }}><div className="fill" style={{ width: `${((rates[c] || 0) / max) * 100}%`, background: COLORS[c] }} /></div>
              <span className="mono small" style={{ textAlign: 'right', color: rates[c] ? COLORS[c] : 'var(--text-3)' }}>{pct(rates[c] || 0, 1)}</span>
            </div>
          )) : <Empty icon="shield" title="No failures detected">None of the detectors fired in this scope.</Empty>}
          <p className="tiny dim">One answer can trigger several detectors, so rates do not sum to the overall failure rate. Degenerate answers skip the other detectors.</p>
        </Card>
        <Card>
          <CardHead icon="info" iconTone="cyan" title="Detectors" />
          {codes.map(c => (
            <div key={c} className="row top" style={{ padding: '7px 0', borderBottom: '1px solid rgba(46,41,36,.5)' }}>
              <span style={{ color: COLORS[c], marginTop: 1 }}>●</span>
              <div><div className="small" style={{ fontWeight: 600 }}>{d.labels[c]} <span className="tiny dim mono">{c}</span></div><div className="tiny muted">{DESC[c]}</div></div>
            </div>
          ))}
        </Card>
      </div>

      <Card className="mt-16">
        <CardHead icon="trend" title="Failure rates over model versions" sub="Benchmark rate per detector for every checkpoint (hollow = rejected candidate)"
          right={<div className="legend">{codes.map(c => <span key={c}><i style={{ background: COLORS[c] }} />{c.replace('_', ' ')}</span>)}</div>} />
        {anyTrend && versions.length ? (
          <LineChart height={260} yDomain={[0, Math.min(1, Math.max(0.1, ...codes.flatMap(c => (d.trend[c] || []).map(t => t.rate))) * 1.15)]}
            yLabel={v => `${Math.round(v * 100)}%`} xLabel={v => (Number.isInteger(v) ? `v${v}` : '')} minXStep={1}
            tooltip={(p, s) => <><div className="tip-title">{s.label}</div><div className="row between"><span className="dim">{p.meta.model}</span><span className="mono">{pct(p.y, 1)}</span></div><div className="tiny dim">{p.meta.status}</div></>}
            series={codes.map(c => ({ key: c, label: d.labels[c], color: COLORS[c], width: 2,
              points: (d.trend[c] || []).filter(t => vIdx[t.model] !== undefined).map(t => ({ x: vIdx[t.model], y: t.rate, meta: t, hollow: t.status === 'rejected' })) }))} />
        ) : <Empty icon="trend" title="Not enough versions yet" />}
      </Card>

      <Card className="mt-16">
        <CardHead icon="search" title="Examples" sub="The most recent answer that triggered each detector" />
        {Object.keys(d.examples || {}).length ? (
          <div className="grid g3">
            {Object.entries(d.examples).map(([c, ex]) => (
              <div key={c} className="card elev hover" style={{ padding: 14, borderColor: `${COLORS[c]}44` }}>
                <div className="row between"><Badge tone="red">{d.labels[c]}</Badge><span className="tiny dim mono">{ex.model}</span></div>
                <div className="small muted mt-8">{ex.persona}</div>
                <div className="response-box mt-8" style={{ fontSize: 13, padding: '10px 14px' }}>{ex.response || <span className="dim">(empty)</span>}</div>
                <div className="tiny mono mt-8" style={{ color: COLORS[c] }}>{ex.detail}</div>
              </div>
            ))}
          </div>
        ) : <Empty icon="shield" title="No flagged answers yet" />}
      </Card>
      <p className="tiny dim mt-12" style={{ cursor: 'pointer' }} onClick={() => nav('dna')}>Per-persona drift is on the Twin DNA page →</p>
    </>)}
  </>)
}
