import { useEffect, useState } from 'react'
import Icon from '../icons.jsx'
import { AXES, AXIS_LABEL, fmt, useApi } from '../api.js'
import { DriftPlot, Radar } from '../charts.jsx'
import { Badge, Card, CardHead, Empty, ErrorBox, Field, Loading, Note, PageHead } from '../ui.jsx'
import { PersonaPicker } from '../components.jsx'

const TRAIT_AXES = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'political']
const LABELS = { ...AXIS_LABEL, style: 'Communication' }

// Radar values: positions (0-10) for the six trait axes; the Communication axis shows the style-match score.
function dnaValues(dna, which) {
  if (!dna) return null
  if (which === 'intended') return { ...dna.intended, style: 10 }
  if (!dna.observed) return null
  return { ...dna.observed, style: dna.axes?.style ?? 0 }
}

export default function TwinDNA({ params }) {
  const models = useApi('/models')
  const personas = useApi('/personas')
  const [model, setModel] = useState(params?.model || null)
  const [pid, setPid] = useState(params?.persona || 'p1')
  useEffect(() => { if (models.data && !model) setModel(models.data.active) }, [models.data]) // eslint-disable-line react-hooks/exhaustive-deps
  const baseline = models.data?.models?.find(m => m.status === 'baseline')?.name
  const dna = useApi(model && pid ? `/metrics/dna?model=${encodeURIComponent(model)}&persona_id=${pid}` : null)
  const base = useApi(baseline && pid && baseline !== model ? `/metrics/dna?model=${encodeURIComponent(baseline)}&persona_id=${pid}` : null)

  const d = dna.data, b = baseline === model ? d : base.data
  const series = [
    { label: 'Intended', color: '#EDE8E2', values: dnaValues(d, 'intended'), dashed: true },
    ...(baseline && baseline !== model ? [{ label: baseline, color: '#7FA7B8', values: dnaValues(b, 'obs'), fillOpacity: 0.06 }] : []),
    { label: model, color: '#E8772E', values: dnaValues(d, 'obs'), fillOpacity: 0.2 },
  ]
  return (<>
    <PageHead eyebrow={<><Icon name="dna" size={13} />Intelligence</>} title="Twin DNA"
      sub="What the persona specifies versus what the model actually expresses, averaged over every scored answer of that model for that persona."
      right={models.data && (
        <Field><select className="select" style={{ width: 260 }} value={model || ''} onChange={e => setModel(e.target.value)}>
          {models.data.models.map(m => <option key={m.name} value={m.name}>{m.name} · {m.status}</option>)}
        </select></Field>
      )} />
    {models.data && !models.data.models.length && <Card><Empty icon="dna" title="No model trained yet">Create the baseline model from the Dashboard first.</Empty></Card>}
    <div className="grid g-main">
      <Card glow="v" className="col" style={{ alignItems: 'center' }}>
        <div className="row between" style={{ width: '100%' }}>
          <span className="upper violet">Behavioral fingerprint</span>
          {d && <span className="tiny dim mono">{d.n} answers · {d.valid ?? 0} measurable</span>}
        </div>
        <ErrorBox error={dna.error} />
        {dna.loading && !d && <div style={{ width: '100%', padding: 40 }}><Loading rows={5} /></div>}
        {d && (<>
          <Radar axes={AXES} labels={LABELS} size={460} center={['TWIN', 'DNA']} spin series={series} />
          <div className="legend">
            <span><i className="dashed" style={{ color: '#EDE8E2' }} />Intended</span>
            {baseline && baseline !== model && <span><i style={{ background: '#7FA7B8' }} />{baseline} (baseline)</span>}
            <span><i style={{ background: '#E8772E' }} />{model}</span>
          </div>
          {!d.observed && <div className="mt-12" style={{ width: '100%' }}><Note tone="warn">{d.n ? 'Every answer of this model for this persona was degenerate, so no traits could be observed.' : 'This model has no evaluated answers for this persona yet. Run a study or the benchmark.'}</Note></div>}
          <p className="tiny dim mt-8" style={{ textAlign: 'center', maxWidth: 520 }}>Trait and politics axes show positions on 0–10. The Communication axis shows the style-match score (10 = the expected style).</p>
        </>)}
      </Card>
      <div className="col gap-16">
        <Card>
          <CardHead icon="user" title="Persona" />
          {personas.data && <PersonaPicker personas={personas.data.personas} value={pid} onChange={setPid} />}
        </Card>
        {d?.observed && (
          <Card>
            <CardHead icon="compare" title="Drift" sub={`○ intended → ● ${model}`} />
            <DriftPlot axes={TRAIT_AXES} intended={d.intended} observed={d.observed} />
          </Card>
        )}
      </div>
    </div>

    {d && (
      <div className="grid g2 mt-16">
        <Card>
          <CardHead icon="target" iconTone="cyan" title="Axis comparison" sub="Positions (0–10) and fidelity per axis" />
          <div className="table-wrap"><table className="t">
            <thead><tr><th>Axis</th><th className="r">Intended</th>{baseline !== model && <th className="r">{baseline}</th>}<th className="r">{model}</th><th className="r">Drift</th><th className="r">Fidelity</th></tr></thead>
            <tbody>
              {TRAIT_AXES.map(a => {
                const dr = d.drift?.[a]
                return (
                  <tr key={a}><td>{LABELS[a]}</td><td className="r mono">{fmt(d.intended[a], 1)}</td>
                    {baseline !== model && <td className="r mono cyan">{fmt(b?.observed?.[a], 1)}</td>}
                    <td className="r mono violet">{fmt(d.observed?.[a], 1)}</td>
                    <td className={`r mono ${dr === undefined || dr === null ? '' : Math.abs(dr) < 1 ? 'green' : Math.abs(dr) < 2.5 ? 'amber' : 'red'}`}>{dr === undefined || dr === null ? '–' : `${dr >= 0 ? '+' : ''}${fmt(dr, 2)}`}</td>
                    <td className="r mono">{fmt(d.axes?.[a], 2)}</td></tr>
                )
              })}
              <tr><td>Communication</td><td className="r dim">style</td>{baseline !== model && <td className="r mono cyan">{fmt(b?.axes?.style, 1)}</td>}<td className="r mono violet">{fmt(d.axes?.style, 1)}</td><td /><td className="r mono">{fmt(d.axes?.style, 2)}</td></tr>
            </tbody>
          </table></div>
          <div className="row between mt-12 small"><span className="muted">Mean fidelity of these answers</span><b className="mono">{fmt(d.overall, 2)}</b></div>
        </Card>
        <Card>
          <CardHead icon="chat" title="Communication style" sub={`Expected from "${d.persona.commStyle || '–'}" vs observed`} />
          {d.style?.expected ? Object.keys(d.style.expected).map((k, i) => {
            const e = d.style.expected[k] ?? 0, o = d.style.observed?.[k] ?? 0
            return (
              <div key={k} style={{ margin: '10px 0', '--i': i }}>
                <div className="row between small"><span className="muted" style={{ textTransform: 'capitalize' }}>{k}</span><span className="mono tiny"><span className="dim">exp</span> {fmt(e, 2)} · <span className="violet">obs</span> {fmt(o, 2)}</span></div>
                <div className="track" style={{ position: 'relative', height: 6, borderRadius: 4, background: 'rgba(237,232,226,.1)', marginTop: 4 }}>
                  <div className="fill" style={{ position: 'absolute', inset: 0, width: `${o * 100}%`, borderRadius: 4, background: '#E8772E', animation: 'grow-x .9s both', transformOrigin: 'left' }} />
                  <span style={{ position: 'absolute', top: -3, bottom: -3, left: `calc(${e * 100}% - 1px)`, width: 2, background: '#7FA7B8' }} title="expected" />
                </div>
              </div>
            )
          }) : <Empty icon="chat" title="No style data" />}
          <div className="tiny dim">Cyan marker = expected share · violet bar = observed share</div>
          {d.persona.is_benchmark ? <div className="mt-8"><Badge tone="cyan">benchmark persona</Badge></div> : null}
        </Card>
      </div>
    )}
  </>)
}
