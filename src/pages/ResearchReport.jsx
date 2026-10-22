import { DriftBars, LineChart, LossChart } from '../charts/charts.jsx'
import { ErrorNote, Loading, Page, Panel, Tag } from '../components/ui.jsx'
import { AXES, fmt, pct, when } from '../constants.js'
import { api } from '../api/client.js'
import { useApi } from '../hooks/useApi.js'

const TITLES = {
  architecture: 'Model architecture', tokenizer: 'Tokenizer', training: 'Training process', data: 'Training data',
  persona_conditioning: 'Persona conditioning', scenarios: 'Scenarios', evaluation: 'Evaluation', feedback_loop: 'Feedback loop',
  benchmark: 'Benchmark', comparison: 'Baseline vs adaptive twin',
}

export default function ResearchReport() {
  const { data: r, error, loading, reload } = useApi('/api/report')
  const runs = useApi('/api/training/runs')
  const sys = useApi('/api/system')
  const firstPersonaRun = runs.data?.runs?.find((x) => x.kind === 'persona')
  const runDetail = useApi(firstPersonaRun ? `/api/training/runs/${firstPersonaRun.id}` : null)
  if (loading && !r) return <Page title="Research report"><Loading /></Page>
  if (error) return <Page title="Research report"><ErrorNote error={error} onRetry={reload} /></Page>
  const models = r.results.models.filter((m) => m.fidelity != null)
  const base = models.find((m) => m.lineage === 'baseline')
  const labels = models.map((m) => `v${m.version}`)
  const active = r.results.models.find((m) => m.status === 'active') || base
  const cal = sys.data?.evaluator?.calibration
  return (
    <Page title="Research report" intro={`Generated ${when(r.generated_at)} from the database. Every figure below is measured; the conclusion is derived from those figures by fixed rules.`}
      actions={<a className="btn primary" href={api.url('/api/export/report.md')}>Download report (Markdown)</a>}>
      <article className="report panel" style={{ padding: '26px 30px' }}>
        <h2 style={{ marginTop: 0 }}>Research question</h2>
        <p style={{ fontFamily: 'var(--serif)', fontSize: 19 }}>{r.research_question}</p>

        <h2>Methodology</h2>
        {Object.entries(r.methodology).map(([k, v]) => (
          <p key={k} style={{ marginBottom: 10 }}><strong>{TITLES[k] || k}.</strong> {v}</p>
        ))}
        {cal && (
          <p style={{ marginBottom: 10 }}><strong>Reading the fidelity scale.</strong> On the benchmark items the evaluator gives {fmt(cal.oracle)} to
            template responses written for the right persona, {fmt(cal.bland)} to a non-committal answer that expresses nothing,
            {' '}{fmt(cal.shuffled)} to answers written for the wrong persona and 0 to unreadable text. A twin is only expressing the persona when it scores above {fmt(cal.bland)}.</p>
        )}

        <h2>Results</h2>
        <div className="scroll">
          <table className="t">
            <thead><tr><th>Model</th><th>Lineage</th><th>Update</th><th>Status</th><th className="num">Fidelity</th><th className="num">Stability</th><th className="num">Consistency</th><th className="num">Failure rate</th><th className="num">Val loss</th><th className="num">Feedback examples</th></tr></thead>
            <tbody>{r.results.models.map((m) => (
              <tr key={m.model}><td>{m.model}</td><td>{m.lineage}</td><td className="faint">{m.kind}</td><td><Tag kind={m.status === 'active' ? 'good' : m.status === 'rejected' ? 'bad' : ''}>{m.status}</Tag></td>
                <td className="num">{fmt(m.fidelity, 2)}</td><td className="num">{fmt(m.stability)}</td><td className="num">{fmt(m.consistency)}</td>
                <td className="num">{pct(m.failure_rate, 1)}</td><td className="num">{fmt(m.val_loss, 3)}</td><td className="num">{m.training_examples}</td></tr>))}</tbody>
          </table>
        </div>
      </article>

      {models.length > 0 && (
        <div className="grid g2">
          <Panel title="Fidelity vs model version"><LineChart labels={labels} series={[
            { name: 'Version', color: 'var(--twin)', points: models.map((m) => ({ y: m.fidelity, hollow: m.status === 'rejected' })) },
            { name: 'Baseline', color: 'var(--persona)', dashed: true, points: models.map(() => ({ y: base?.fidelity })) }]} /></Panel>
          <Panel title="Stability vs model version"><LineChart labels={labels} series={[{ name: 'Stability', color: 'var(--twin)', points: models.map((m) => ({ y: m.stability })) }]} /></Panel>
          <Panel title="Failure rate vs model version"><LineChart labels={labels} format={(v) => `${(v * 100).toFixed(0)}%`} series={[{ name: 'Failure rate', color: 'var(--bad)', points: models.map((m) => ({ y: m.failure_rate })) }]} /></Panel>
          <Panel title="Validation loss vs model version"><LineChart labels={labels} format={(v) => v.toFixed(3)} series={[{ name: 'Validation loss', color: 'var(--persona)', points: models.map((m) => ({ y: m.val_loss })) }]} /></Panel>
          <Panel title="Training loss (stage 2, persona specialisation)">{runDetail.data ? <LossChart log={runDetail.data.log} /> : <p className="faint">No training run.</p>}</Panel>
          <Panel title={`Per-axis gap to a perfect score (${active?.model || '—'})`}>
            {active?.axes && Object.keys(active.axes).length ? (
              <DriftBars rows={AXES.map((a) => ({ label: a.l, value: (active.axes[a.k] ?? 10) - 10 }))} span={10} />
            ) : <p className="faint">No benchmark.</p>}
            <p className="faint" style={{ marginTop: 6 }}>Bars show each axis score minus a perfect 10.</p>
          </Panel>
        </div>
      )}

      <article className="report panel" style={{ padding: '26px 30px' }}>
        <h2 style={{ marginTop: 0 }}>Conclusion</h2>
        <ul>{r.conclusion.map((c, i) => <li key={i} style={{ marginBottom: 8 }}>{c}</li>)}</ul>
        <h2>Status of claims</h2>
        <table className="t" style={{ maxWidth: 720 }}>
          <tbody>{r.status.map((s) => <tr key={s.item}><td>{s.item}</td><td><Tag kind={/^Implemented|^Trained|^Evaluated/.test(s.state) ? 'good' : /Hypothetical|Placeholder/.test(s.state) ? 'warn' : ''}>{s.state}</Tag></td></tr>)}</tbody>
        </table>
      </article>
    </Page>
  )
}
