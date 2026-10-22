import { useEffect, useState } from 'react'
import { api } from '../api/client.js'
import { Btn, ErrorNote, FailureTags, Loading, Page, Panel, StatusTag, Tag } from '../components/ui.jsx'
import { fmt, pct, signed, when } from '../constants.js'
import { useApi, useJob } from '../hooks/useApi.js'
import { useLab } from '../state.jsx'

function Detail({ name, onChanged }) {
  const { data: m, error, reload } = useApi(`/api/models/${name}`)
  const lab = useLab()
  const [err, setErr] = useState(null)
  const [jobId, setJobId] = useState(null)
  const job = useJob(jobId, () => { reload(); onChanged() })
  if (error) return <ErrorNote error={error} />
  if (!m) return <Loading />
  const isActive = lab.active === m.name
  const activeV = lab.models.find((x) => x.name === lab.active)?.version
  const activate = async (force = false) => {
    setErr(null)
    try { await api.post(`/api/models/${m.name}/activate`, { force }); await lab.refresh(); onChanged() } catch (e) {
      if (e.status === 403 && window.confirm(`${e.message}\n\nActivate anyway?`)) return activate(true)
      setErr(e)
    }
  }
  const dec = m.decision
  const meta = m.checkpoint.meta || {}
  return (
    <Panel title={m.name} actions={<>
      {!isActive && <Btn small kind="primary" onClick={() => activate()}>{m.version < activeV ? 'Roll back to this version' : 'Activate'}</Btn>}
      <Btn small onClick={async () => { try { setJobId((await api.post(`/api/models/${m.name}/benchmark`)).job_id) } catch (e) { setErr(e) } }} disabled={job && job.status === 'running'}>Re-run benchmark</Btn>
    </>}>
      <div className="stack">
        <ErrorNote error={err} />
        {job && <p className="faint">Benchmark job: {job.status} {job.message}</p>}
        <div className="row"><StatusTag status={m.lineage === 'baseline' ? 'baseline' : m.status} />{isActive && <Tag kind="twin">active</Tag>}<span className="faint">{m.notes}</span></div>
        {dec && (
          <div className={`banner ${dec.accepted ? 'good' : 'bad'}`}>
            <div className="stack" style={{ gap: 3 }}>
              <strong>{dec.accepted ? 'Accepted' : 'Rejected'}: fidelity {fmt(dec.previous_score)} → {fmt(dec.new_score)} ({signed(dec.improvement, 2)})</strong>
              {dec.reasons.map((r, i) => <span key={i} className="faint">{r}</span>)}
            </div>
          </div>
        )}
        <dl className="kv">
          <dt>Parent</dt><dd>{m.parent || '— (random initialisation)'}</dd>
          <dt>Created</dt><dd>{when(m.created_at)}</dd>
          <dt>Parameters</dt><dd>{m.parameters.toLocaleString()}</dd>
          <dt>Architecture</dt><dd>{m.model_config.num_layers} layers · {m.model_config.num_heads} heads · embedding {m.model_config.embedding_dim} · context {m.model_config.context_length} · vocab {m.model_config.vocab_size}</dd>
          <dt>Backend</dt><dd>{m.backend}</dd>
          <dt>Tokenizer</dt><dd>{m.tokenizer_version}</dd>
          <dt>Code version</dt><dd>{m.code_version}</dd>
          <dt>Dataset</dt><dd>{m.dataset_id}{m.dataset ? ` — ${m.dataset.description} (${m.dataset.n_examples} examples, sha ${m.dataset.sha})` : ''}</dd>
          <dt>Weights sha256</dt><dd>{meta.weights_sha256}</dd>
          <dt>Checkpoint</dt><dd>{m.checkpoint.path} {m.checkpoint.exists ? '' : '(missing!)'}<br /><span className="faint">{m.checkpoint.files.join(', ')}</span></dd>
          <dt>Training run</dt><dd>{m.training_run ? `#${m.training_run.id} ${m.training_run.kind}: ${m.training_run.steps} steps, val loss ${fmt(m.training_run.final_val_loss, 3)}` : '—'}</dd>
        </dl>
        <details><summary>Training configuration</summary><pre className="code">{JSON.stringify(m.training_config, null, 2)}</pre></details>
        <details><summary>Benchmark history ({m.metrics_history.length})</summary>
          <table className="t"><thead><tr><th>When</th><th className="num">Fidelity</th><th className="num">Stability</th><th className="num">Failure rate</th></tr></thead>
            <tbody>{m.metrics_history.map((h) => <tr key={h.id}><td className="faint">{when(h.created_at)}</td><td className="num">{fmt(h.fidelity, 2)}</td><td className="num">{fmt(h.stability)}</td><td className="num">{pct(h.failure_rate, 1)}</td></tr>)}</tbody></table>
        </details>
      </div>
    </Panel>
  )
}

function Compare({ models }) {
  const [a, setA] = useState('')
  const [b, setB] = useState('')
  useEffect(() => {
    if (models.length >= 2 && !a) { setA(models[0].name); setB(models[models.length - 1].name) }
  }, [models])
  const { data, error } = useApi(a && b && a !== b ? `/api/models/compare/${a}/${b}` : null)
  const row = (label, k, f = (v) => fmt(v, 2), better = 'high') => {
    const va = data?.a?.[k], vb = data?.b?.[k]
    const d = va != null && vb != null ? vb - va : null
    const good = d == null ? null : better === 'high' ? d > 0 : d < 0
    return <tr key={k}><td>{label}</td><td className="num">{va == null ? '—' : f(va)}</td><td className="num">{vb == null ? '—' : f(vb)}</td>
      <td className="num" style={{ color: d == null || d === 0 ? undefined : good ? 'var(--good)' : 'var(--bad)' }}>{d == null ? '—' : (d > 0 ? '+' : '') + f(d)}</td></tr>
  }
  return (
    <Panel title="Compare two versions">
      <div className="row" style={{ marginBottom: 12 }}>
        <select value={a} onChange={(e) => setA(e.target.value)} style={{ width: 180 }}>{models.map((m) => <option key={m.name}>{m.name}</option>)}</select>
        <span className="faint">vs</span>
        <select value={b} onChange={(e) => setB(e.target.value)} style={{ width: 180 }}>{models.map((m) => <option key={m.name}>{m.name}</option>)}</select>
      </div>
      <ErrorNote error={error} />
      {data && (
        <div className="stack">
          <table className="t" style={{ maxWidth: 620 }}>
            <thead><tr><th>Metric</th><th className="num">{a}</th><th className="num">{b}</th><th className="num">Δ</th></tr></thead>
            <tbody>
              {row('Fidelity', 'fidelity')}
              {row('Stability', 'stability')}
              {row('Consistency', 'consistency')}
              {row('Failure rate', 'failure_rate', (v) => pct(v, 1), 'low')}
              {row('Validation loss', 'val_loss', (v) => v.toFixed(3), 'low')}
              {row('Mean words / response', 'avg_length', (v) => v.toFixed(1), 'none')}
            </tbody>
          </table>
          <h3>Same prompt, same seed ({data.n_pairs} benchmark items, largest gains first)</h3>
          {data.pairs.length === 0 && <p className="sub">Both versions need a benchmark run.</p>}
          {data.pairs.map((p, i) => (
            <div key={i} className="panel" style={{ background: 'var(--bg)' }}>
              <p className="faint" style={{ marginBottom: 8 }}>{p.persona} · {p.domain} · seed {p.seed} — {p.scenario.slice(0, 120)}</p>
              <div className="grid g2">
                <div className="stack" style={{ gap: 6 }}><span className="sub">{a}: {fmt(p.a.fidelity)}</span><div className="response small" style={{ borderColor: 'var(--persona)' }}>{p.a.response || '(empty)'}</div><FailureTags codes={p.a.failures} empty="none" /></div>
                <div className="stack" style={{ gap: 6 }}><span className="sub">{b}: {fmt(p.b.fidelity)}</span><div className="response small">{p.b.response || '(empty)'}</div><FailureTags codes={p.b.failures} empty="none" /></div>
              </div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  )
}

const kindOf = (m) => (m.lineage === 'baseline' ? 'bootstrap' : (m.notes || '').split(' ')[0])

export default function ModelRegistry() {
  const lab = useLab()
  const { data, reload } = useApi('/api/models')
  const [sel, setSel] = useState(null)
  const models = data?.models || []
  useEffect(() => { if (!sel && models.length) setSel(data.active || models[0].name) }, [data])
  return (
    <Page title="Model Registry" intro="Every checkpoint ever trained, with the data, code and decision that produced it. Nothing is deleted automatically.">
      {!models.length ? <p className="sub">No model versions yet.</p> : (
        <>
          <Panel title="Versions">
            <div className="scroll">
              <table className="t">
                <thead><tr><th>Model</th><th>Lineage</th><th>Update</th><th>Status</th><th>Created</th><th className="num">Fidelity</th><th className="num">Stability</th><th className="num">Failure rate</th><th className="num">Val loss</th><th>Dataset</th></tr></thead>
                <tbody>
                  {models.map((m) => (
                    <tr key={m.name} className={`clickable ${sel === m.name ? 'selected' : ''}`} onClick={() => setSel(m.name)}>
                      <td>{m.name} {m.is_active && <Tag kind="twin">active</Tag>}</td><td>{m.lineage}</td><td className="faint">{kindOf(m)}</td><td><StatusTag status={m.lineage === 'baseline' ? 'baseline' : m.status} /></td>
                      <td className="faint">{when(m.created_at)}</td><td className="num">{fmt(m.metrics?.fidelity, 2)}</td><td className="num">{fmt(m.metrics?.stability)}</td>
                      <td className="num">{pct(m.metrics?.failure_rate, 1)}</td><td className="num">{fmt(m.metrics?.val_loss, 3)}</td><td className="faint">{m.dataset_id}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          {sel && <Detail key={sel} name={sel} onChanged={() => { reload(); lab.refresh() }} />}
          {models.length >= 2 && <Compare models={models} />}
        </>
      )}
    </Page>
  )
}
