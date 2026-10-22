import Icon from '../icons.jsx'
import { compact, dateTime, fmt, pct, useApi } from '../api.js'
import { Badge, Card, Empty, ErrorBox, Loading, PageHead } from '../ui.jsx'
import { FidelityCurve, PairedLabel } from '../components.jsx'

const EXPORTS = [
  ['report.md', 'Research report', 'Markdown'], ['trials.csv', 'All trials', 'CSV'], ['benchmark.csv', 'Benchmark trials', 'CSV'],
  ['models.json', 'Models & training runs', 'JSON'], ['experiments.json', 'Experiments', 'JSON'],
]
const STATE_TONE = s => (s.startsWith('Implemented') || s.startsWith('Trained') || s.startsWith('Evaluated') ? 'green' : s.startsWith('Placeholder') ? 'amber' : s.startsWith('Hypothetical') ? 'red' : '')
const METHOD_LABEL = { architecture: 'Architecture', tokenizer: 'Tokenizer', training: 'Training', data: 'Data', persona_conditioning: 'Persona conditioning',
  scenarios: 'Scenarios', evaluation: 'Evaluation', feedback_loop: 'Feedback loop', benchmark: 'Benchmark', comparison: 'Comparison' }

export default function Report() {
  const r = useApi('/report')
  const d = r.data
  return (<>
    <PageHead eyebrow={<><Icon name="report" size={13} />Research</>} title="Research Report"
      sub="Generated from the database on request. It states what was measured, including negative results."
      right={<a className="btn primary lg" href="/api/export/report.md" download><Icon name="download" size={16} />EXPORT REPORT</a>} />
    <ErrorBox error={r.error} />
    {r.loading && !d && <Card><Loading rows={8} /></Card>}
    {d && (
      <div className="col gap-16 stagger">
        <Card className="report-sec" glow="v" i={0}>
          <div className="row between wrap"><span className="upper violet">{d.title}</span><span className="tiny dim mono">generated {dateTime(d.generated_at)}</span></div>
          <h2 style={{ fontSize: 21, marginTop: 10 }}>{d.research_question}</h2>
          <div className="row wrap gap-6 mt-12">{d.status.map(s => <Badge key={s.item} tone={`${STATE_TONE(s.state)} wrap`} title={s.item}>{s.item}: {s.state}</Badge>)}</div>
        </Card>
        <Card className="report-sec" i={1}>
          <h2><span className="n">01</span>Methodology</h2>
          <dl>{Object.entries(d.methodology).map(([k, v]) => <FragmentRow key={k} k={METHOD_LABEL[k] || k} v={v} />)}</dl>
        </Card>
        <Card className="report-sec" i={2}>
          <h2><span className="n">02</span>Results</h2>
          {d.results.models.length ? (<>
            <FidelityCurve curve={d.results.models} height={240} compact />
            <div className="table-wrap mt-12">
              <table className="t">
                <thead><tr><th>Model</th><th>Status</th><th>Update</th><th className="r">Fidelity</th><th>vs parent (paired)</th><th className="r">Stability</th><th className="r">Failure</th><th className="r">Val loss</th><th className="r">Params</th></tr></thead>
                <tbody>{d.results.models.map(m => (
                  <tr key={m.model}><td className="mono small">{m.model}</td><td><Badge tone={m.status === 'rejected' ? 'red' : m.status === 'baseline' ? 'cyan' : 'violet'}>{m.status}</Badge></td><td className="small">{m.kind}</td>
                    <td className="r mono">{fmt(m.fidelity, 2)}</td><td className="small">{m.paired_vs_parent ? <PairedLabel p={m.paired_vs_parent} /> : '–'}</td>
                    <td className="r mono">{fmt(m.stability, 1)}</td><td className="r mono">{pct(m.failure_rate, 1)}</td><td className="r mono">{fmt(m.val_loss, 3)}</td><td className="r mono">{compact(m.parameters)}</td></tr>
                ))}</tbody>
              </table>
            </div>
            <div className="row wrap gap-20 small mt-12 muted">
              <span>Benchmark runs <b className="mono">{d.results.benchmark_runs}</b></span>
              <span>Human ratings <b className="mono">{d.results.human_evaluations?.n ?? 0}</b>{d.results.human_evaluations?.mean_score ? ` (mean ${fmt(d.results.human_evaluations.mean_score, 1)}/10)` : ''}</span>
              <span>Training examples <b className="mono">{Object.values(d.results.training_examples || {}).reduce((a, b) => a + b, 0)}</b> {Object.entries(d.results.training_examples || {}).map(([k, v]) => `${k} ${v}`).join(' · ')}</span>
            </div>
          </>) : <Empty icon="report" title="No models yet">Bootstrap TwinSim-v0 to produce results.</Empty>}
        </Card>
        {d.results.learning_experiment && (
          <Card className="report-sec" i={3}>
            <h2><span className="n">03</span>Learning experiment</h2>
            <p>Baseline <b className="mono">{fmt(d.results.learning_experiment.initial_fidelity, 2)}</b> → final <b className="mono">{fmt(d.results.learning_experiment.final_fidelity, 2)}</b> ({d.results.learning_experiment.improvement >= 0 ? '+' : ''}{fmt(d.results.learning_experiment.improvement, 2)}); {d.results.learning_experiment.accepted} accepted, {d.results.learning_experiment.rejected} rejected.</p>
          </Card>
        )}
        <Card className="report-sec" i={4}>
          <h2><span className="n">{d.results.learning_experiment ? '04' : '03'}</span>Failure analysis</h2>
          {Object.keys(d.results.failures?.current || {}).length ? (
            <div className="row wrap gap-6">{Object.entries(d.results.failures.current).map(([k, v]) => <Badge key={k} tone="red">{d.results.failures.labels[k]} {pct(v, 1)}</Badge>)}</div>
          ) : <p>No failure modes on the active model's benchmark.</p>}
        </Card>
        <Card className="report-sec" i={5}>
          <h2><span className="n">{d.results.learning_experiment ? '05' : '04'}</span>Conclusion</h2>
          <ul>{d.conclusion.map((c, i) => <li key={i}>{c}</li>)}</ul>
        </Card>
        <Card className="report-sec" i={6}>
          <h2><span className="n"><Icon name="download" size={12} /></span>Exports</h2>
          <div className="grid g3">
            {EXPORTS.map(([f, l, kind]) => (
              <a key={f} href={`/api/export/${f}`} download className="card elev hover" style={{ padding: 14, color: 'inherit', display: 'flex', alignItems: 'center', gap: 12 }}>
                <span className="m-icon" style={{ width: 34, height: 34, borderRadius: 9, display: 'grid', placeItems: 'center', background: 'var(--accent-soft)', color: 'var(--accent)' }}><Icon name="download" size={16} /></span>
                <div><div style={{ fontWeight: 600 }}>{l}</div><div className="tiny dim mono">{f} · {kind}</div></div>
              </a>
            ))}
          </div>
        </Card>
      </div>
    )}
  </>)
}

function FragmentRow({ k, v }) { return (<><dt>{k}</dt><dd>{v}</dd></>) }
