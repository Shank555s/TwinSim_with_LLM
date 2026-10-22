import { HBars, VersionBars } from '../charts/charts.jsx'
import { ErrorNote, Loading, Page, Panel } from '../components/ui.jsx'
import { FAILURE_HELP, FAILURE_LABELS, pct } from '../constants.js'
import { useApi } from '../hooks/useApi.js'

function trendWord(rows) {
  const v = rows.filter((r) => r.rate != null)
  if (v.length < 2) return null
  const d = v[v.length - 1].rate - v[0].rate
  if (Math.abs(d) < 0.01) return { t: 'flat', c: 'var(--muted)' }
  return d < 0 ? { t: `down ${pct(-d, 1)}`, c: 'var(--good)' } : { t: `up ${pct(d, 1)}`, c: 'var(--bad)' }
}

export default function FailureAnalysis() {
  const { data, error, loading, reload } = useApi('/api/metrics/failures')
  if (loading && !data) return <Page title="Failure Analysis"><Loading /></Page>
  if (error) return <Page title="Failure Analysis"><ErrorNote error={error} onRetry={reload} /></Page>
  const codes = Object.keys(data.labels)
  const cur = data.current || {}
  return (
    <Page title="Failure Analysis" intro="How often each SimFidelity failure mode appears on the fixed benchmark, and whether it rises or falls as the twin is retrained.">
      <div className="grid g2" style={{ alignItems: 'start' }}>
        <Panel title="Active model" hint={data.active_model || ''}>
          <HBars max={1} color="var(--bad)" format={(v) => pct(v, 1)}
            rows={codes.map((c) => ({ label: FAILURE_LABELS[c], value: cur[c] || 0, title: FAILURE_HELP[c] })).sort((a, b) => b.value - a.value)} />
          <p className="faint" style={{ marginTop: 10 }}>Share of benchmark responses showing each failure (a response can show several).</p>
        </Panel>
        <Panel title="All other trials" hint={`${data.all_trials.n} user, study and harvest trials`}>
          <HBars max={1} color="var(--bad)" format={(v) => pct(v, 1)}
            rows={codes.map((c) => ({ label: FAILURE_LABELS[c], value: data.all_trials.rates[c] || 0 })).sort((a, b) => b.value - a.value)} />
        </Panel>
      </div>
      <div className="grid g2">
        {codes.map((c) => {
          const rows = data.trend[c] || []
          const tw = trendWord(rows)
          const ex = data.examples[c]
          return (
            <Panel key={c} title={FAILURE_LABELS[c]} hint={tw ? <span style={{ color: tw.c }}>{tw.t} since v0</span> : ''}>
              <p className="sub" style={{ marginBottom: 10 }}>{FAILURE_HELP[c]}</p>
              <VersionBars rows={rows.map((r) => ({ label: r.model.replace('TwinSim-', ''), value: r.rate, dim: r.status === 'rejected', highlight: r.model === data.active_model }))} />
              {ex && (
                <details style={{ marginTop: 10 }}>
                  <summary>Latest example ({ex.model}, {ex.persona})</summary>
                  <div className="response small" style={{ marginTop: 8, borderColor: 'var(--bad)' }}>{ex.response || '(empty)'}</div>
                  <p className="faint" style={{ marginTop: 6 }}>{ex.detail}</p>
                </details>
              )}
            </Panel>
          )
        })}
      </div>
      <p className="faint">Grey bars are rejected candidates. Exaggeration, blue shift, stereotype and inconsistency are the original SimFidelity taxonomy; red shift and degenerate output were added for the from-scratch model.</p>
    </Page>
  )
}
