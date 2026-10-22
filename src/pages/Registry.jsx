import { useEffect, useState } from 'react'
import Icon from '../icons.jsx'
import { AXIS_LABEL, api, compact, dateTime, fidelityColor, fmt, pct, signed, useApi, useJob } from '../api.js'
import { Badge, Btn, Card, CardHead, CountUp, Empty, ErrorBox, Loading, Modal, PageHead, Progress, Seg, StatusBadge, useApp } from '../ui.jsx'
import { PairedLabel } from '../components.jsx'

export default function Registry({ params }) {
  const [tab, setTab] = useState(params?.tab || 'versions')
  return (<>
    <PageHead eyebrow={<><Icon name="layers" size={13} />Models</>} title="Model Registry"
      sub="Every checkpoint is kept on disk with its parent, dataset, training run and acceptance decision. Any of them can be activated or rolled back to."
      right={<Seg value={tab} onChange={setTab} options={[['versions', 'Versions'], ['compare', 'Baseline vs Adaptive']]} />} />
    <div key={tab} className="fade-in">{tab === 'versions' ? <Versions focus={params?.model} /> : <Compare a0={params?.a} b0={params?.b} />}</div>
  </>)
}

function Versions({ focus }) {
  const { toast, refreshStatus, nav, status } = useApp()
  const pub = !!status?.public_mode
  const list = useApi('/models')
  const curve = useApi('/metrics/learning-curve')
  const [open, setOpen] = useState(focus || null)
  const [confirm, setConfirm] = useState(null)
  const [err, setErr] = useState(null)
  const [benchJob, setBenchJob] = useState(null)
  const bj = useJob(benchJob, { onDone: j => { toast(j.status === 'completed' ? 'Benchmark complete' : `Benchmark ${j.status}`, j.status === 'completed' ? 'ok' : 'err'); list.reload(true); curve.reload(true); refreshStatus() } })
  const cmap = Object.fromEntries((curve.data?.curve || []).map(c => [c.model, c]))

  async function activate(m, force = false) {
    setErr(null)
    try {
      const r = await api(`/models/${m.name}/activate`, { method: 'POST', body: { force } })
      toast(r.changed ? `${r.rollback ? 'Rolled back to' : 'Activated'} ${m.name}` : `${m.name} is already active`, 'ok')
      setConfirm(null); list.reload(true); curve.reload(true); refreshStatus()
    } catch (e) { setErr(e); setConfirm(null) }
  }
  async function bench(m) {
    try { const r = await api(`/models/${m.name}/benchmark`, { method: 'POST' }); setBenchJob(r.job_id); refreshStatus() } catch (e) { setErr(e) }
  }
  if (list.error) return <ErrorBox error={list.error} />
  if (!list.data) return <Card><Loading rows={5} /></Card>
  const models = [...list.data.models].sort((a, b) => b.version - a.version)
  const active = list.data.models.find(m => m.is_active)
  if (!models.length) return <Card><Empty icon="layers" title="No models yet" action={<Btn kind="primary" onClick={() => nav('dashboard')}>Go to Dashboard</Btn>}>Create the baseline model from the Dashboard to add the first entry.</Empty></Card>

  return (<>
    <ErrorBox error={err} />
    {bj && ['queued', 'running'].includes(bj.status) && (
      <Card glow="c" className="mb-16" style={{ marginBottom: 16 }}>
        <div className="row between small"><span><b>Running research benchmark…</b> <span className="muted">{bj.message}</span></span><span className="mono">{Math.round((bj.progress || 0) * 100)}%</span></div>
        <div className="mt-8"><Progress value={bj.progress} /></div>
      </Card>
    )}
    <div className="lineage-rail col gap-16 stagger">
      {models.map((m, i) => {
        const c = cmap[m.name] || {}
        const met = m.metrics || {}
        const isOpen = open === m.name
        return (
          <div key={m.name} className={`card flush ${m.status} ${m.is_active ? 'glow-v' : ''}`} style={{ '--i': i }}>
            <div className={`vcard ${m.is_active ? 'active' : ''} ${m.status}`}>
              <div className="ver">v{m.version}</div>
              <div style={{ minWidth: 0 }}>
                <div className="row wrap gap-6">
                  <span style={{ fontWeight: 650, fontSize: 15 }} className="mono">{m.name}</span>
                  <StatusBadge status={m.status} />
                  {m.is_active && m.status !== 'active' && <Badge tone="solid-v" upper>active</Badge>}
                  <Badge>{m.kind === 'auto' ? 'automatic' : m.kind}</Badge>
                  {!!m.weights_pruned && <Badge tone="amber" title="Weights removed by the checkpoint retention policy; metrics and decision are kept">weights pruned</Badge>}
                  {m.parent && <span className="tiny dim">from <span className="mono">{m.parent}</span></span>}
                  <span className="tiny dim">· {dateTime(m.created_at)}</span>
                </div>
                <div className="vmetrics">
                  <div className="vmetric"><div className="k">Fidelity</div><div className="v" style={{ color: fidelityColor(met.fidelity) }}>{fmt(met.fidelity, 2)}</div></div>
                  <div className="vmetric"><div className="k">vs parent (paired)</div><div className="v">{c.paired_vs_parent ? <PairedLabel p={c.paired_vs_parent} /> : '–'}</div></div>
                  <div className="vmetric"><div className="k">Stability</div><div className="v">{fmt(met.stability, 1)}</div></div>
                  <div className="vmetric"><div className="k">Failure rate</div><div className="v">{pct(met.failure_rate, 1)}</div></div>
                  <div className="vmetric"><div className="k">Val loss</div><div className="v">{fmt(c.val_loss, 3)}</div></div>
                  <div className="vmetric"><div className="k">Params</div><div className="v">{compact(m.parameters)}</div></div>
                </div>
              </div>
              <div className="col gap-6" style={{ alignItems: 'flex-end' }}>
                {!m.is_active && !pub && !m.weights_pruned && <Btn size="sm" kind={m.status === 'rejected' ? '' : 'primary'} icon={active && m.version < active.version ? 'rollback' : 'bolt'}
                  onClick={() => (m.status === 'rejected' ? setConfirm(m) : activate(m))}>{active && m.version < active.version ? 'Roll back' : 'Activate'}</Btn>}
                <div className="row gap-6">
                  <Btn size="sm" kind="ghost" icon="gauge" title="Re-run the fixed benchmark" disabled={pub || !!m.weights_pruned || (bj && ['queued', 'running'].includes(bj.status))} onClick={() => bench(m)}>Benchmark</Btn>
                  <Btn size="sm" kind="ghost" icon={isOpen ? 'chevronD' : 'chevronR'} onClick={() => setOpen(isOpen ? null : m.name)}>Details</Btn>
                </div>
              </div>
            </div>
            {isOpen && <Details name={m.name} />}
          </div>
        )
      })}
    </div>
    <Modal open={!!confirm} onClose={() => setConfirm(null)} width={520}>
      {confirm && (<>
        <h3 className="card-title"><Icon name="warn" size={16} className="amber" />Activate a rejected candidate?</h3>
        <p className="small muted">{confirm.name} failed the acceptance rule:</p>
        <ul className="list-plain small muted">{(confirm.decision?.reasons || []).map((r, i) => <li key={i}>{r}</li>)}</ul>
        <p className="small muted">Activating it anyway is recorded in the history log as forced.</p>
        <div className="row mt-16"><Btn kind="danger" onClick={() => activate(confirm, true)}>Activate anyway</Btn><Btn kind="ghost" onClick={() => setConfirm(null)}>Cancel</Btn></div>
      </>)}
    </Modal>
  </>)
}

function Details({ name }) {
  const d = useApi(`/models/${name}`)
  const m = d.data
  return (
    <div style={{ borderTop: '1px solid var(--border)', padding: '14px 18px 18px' }} className="fade-in">
      {!m ? <Loading rows={3} /> : (
        <div className="grid g3">
          <div>
            <div className="section-label">Architecture</div>
            <dl className="kv" style={{ gridTemplateColumns: '110px 1fr' }}>
              <dt>Layers × heads</dt><dd className="mono">{m.model_config.num_layers} × {m.model_config.num_heads}</dd>
              <dt>Embedding</dt><dd className="mono">{m.model_config.embedding_dim}</dd>
              <dt>Context</dt><dd className="mono">{m.model_config.context_length}</dd>
              <dt>Vocabulary</dt><dd className="mono">{m.model_config.vocab_size}</dd>
              <dt>Parameters</dt><dd className="mono">{m.parameters.toLocaleString('en-US')}</dd>
              <dt>Backend</dt><dd className="mono">{m.backend}</dd>
              <dt>Tokenizer</dt><dd className="mono">{m.tokenizer_version}</dd>
              <dt>Code</dt><dd className="mono">{m.code_version}</dd>
            </dl>
          </div>
          <div>
            <div className="section-label">Training</div>
            {m.training_run ? (
              <dl className="kv" style={{ gridTemplateColumns: '110px 1fr' }}>
                <dt>Run</dt><dd className="mono">#{m.training_run.id} · {m.training_run.kind}</dd>
                <dt>Steps</dt><dd className="mono">{m.training_run.steps}</dd>
                <dt>Tokens</dt><dd className="mono">{compact(m.training_run.tokens_seen)}</dd>
                <dt>Train / val</dt><dd className="mono">{fmt(m.training_run.final_train_loss, 3)} / {fmt(m.training_run.final_val_loss, 3)}</dd>
                <dt>Dataset</dt><dd className="mono">{m.dataset?.name || m.dataset_id}</dd>
                <dt>LR</dt><dd className="mono">{m.training_config?.learning_rate}</dd>
              </dl>
            ) : <span className="small dim">no run recorded</span>}
            <div className="section-label mt-12">Checkpoint</div>
            <div className="tiny mono dim" style={{ wordBreak: 'break-all' }}>{m.checkpoint.path}</div>
            <div className="row wrap gap-4 mt-4">{m.checkpoint.files.map(f => <Badge key={f}>{f}</Badge>)}</div>
            {!m.checkpoint.exists && <Badge tone="red">missing on disk</Badge>}
          </div>
          <div>
            <div className="section-label">Acceptance decision</div>
            {m.decision ? (<ul className="list-plain small muted">{m.decision.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>) : <span className="small dim">{m.lineage === 'baseline' ? 'Baseline: no acceptance rule (trained from scratch).' : 'no decision recorded'}</span>}
            <div className="section-label mt-12">Benchmarks run</div>
            <span className="mono small">{m.metrics_history.length}</span>
          </div>
        </div>
      )}
    </div>
  )
}

function Compare({ a0, b0 }) {
  const list = useApi('/models')
  const [a, setA] = useState(a0 || null)
  const [b, setB] = useState(b0 || null)
  useEffect(() => {
    if (!list.data) return
    if (!a) setA(list.data.models.find(m => m.status === 'baseline')?.name)
    if (!b) setB(list.data.active)
  }, [list.data]) // eslint-disable-line react-hooks/exhaustive-deps
  const c = useApi(a && b ? `/models/compare/${a}/${b}` : null)
  const d = c.data
  const opts = list.data?.models || []
  const delta = d?.a && d?.b ? d.b.fidelity - d.a.fidelity : null
  const p = d?.paired
  return (<>
    <Card>
      <div className="row wrap gap-16">
        <div className="row"><Badge tone="cyan">A</Badge><select className="select" style={{ width: 240 }} value={a || ''} onChange={e => setA(e.target.value)}>{opts.map(m => <option key={m.name} value={m.name}>{m.name} · {m.status}</option>)}</select></div>
        <Icon name="compare" size={16} className="dim" />
        <div className="row"><Badge tone="violet">B</Badge><select className="select" style={{ width: 240 }} value={b || ''} onChange={e => setB(e.target.value)}>{opts.map(m => <option key={m.name} value={m.name}>{m.name} · {m.status}</option>)}</select></div>
        <span className="small dim">Both models answered the identical benchmark items with identical seeds.</span>
      </div>
    </Card>
    <ErrorBox error={c.error} />
    {c.loading && !d && <Card className="mt-16"><Loading rows={4} /></Card>}
    {d && d.a && d.b && (<>
      <div className="vs mt-16">
        <ModelSide m={d.a} tone="cyan" label="A" />
        <div className="vs-center pop-in">
          <span className="upper dim">Δ fidelity (B − A)</span>
          <div className="delta-big" style={{ color: delta > 0 ? '#5FB37A' : delta < 0 ? '#D96459' : 'var(--text)' }}><CountUp value={delta} decimals={2} prefix={delta > 0 ? '+' : ''} /></div>
          {p ? (<>
            <div className="small mono">± {fmt(p.se, 2)} SE · z {fmt(p.z, 2)}</div>
            <Badge tone={Math.abs(p.z) >= 2 ? (p.mean > 0 ? 'green' : 'red') : ''}>{Math.abs(p.z) >= 2 ? 'beyond noise (|z| ≥ 2)' : 'within noise'}</Badge>
            <div className="tiny dim">{p.n} paired items</div>
          </>) : <div className="tiny dim">no paired benchmark data</div>}
        </div>
        <ModelSide m={d.b} tone="violet" label="B" />
      </div>
      <Card className="mt-16">
        <CardHead icon="target" title="Axis deltas" sub="Mean benchmark axis score (0–10), B − A" />
        <div className="grid g4">
          {Object.keys(d.b.axes || {}).map((k, i) => {
            const dv = (d.b.axes[k] ?? 0) - (d.a.axes?.[k] ?? 0)
            return (
              <div key={k} className="card elev pop-in" style={{ padding: 12, animationDelay: `${i * 50}ms` }}>
                <div className="tiny dim">{AXIS_LABEL[k]}</div>
                <div className="row between mt-4"><span className="mono small">{fmt(d.a.axes?.[k], 2)} → {fmt(d.b.axes[k], 2)}</span>
                  <span className={`mono ${dv > 0.05 ? 'green' : dv < -0.05 ? 'red' : 'dim'}`}>{signed(dv, 2)}</span></div>
              </div>
            )
          })}
        </div>
      </Card>
      <Card className="mt-16">
        <CardHead icon="chat" title="Paired answers" sub={`Largest improvements first · showing ${d.pairs.length} of ${d.n_pairs} (repetition 0)`} />
        {d.pairs.length ? d.pairs.map((pr, i) => (
          <div key={i} className="grid g2" style={{ padding: '12px 0', borderTop: i ? '1px solid var(--border)' : 0 }}>
            <div className="col gap-4" style={{ gridColumn: '1 / -1' }}><span className="small"><b>{pr.persona}</b> <span className="dim">·</span> <span className="muted">{pr.scenario}</span></span></div>
            {[['A', pr.a, 'cyan'], ['B', pr.b, 'violet']].map(([l, x, t]) => (
              <div key={l} className="card elev" style={{ padding: 12 }}>
                <div className="row between"><Badge tone={t}>{l}</Badge><span className="mono" style={{ color: fidelityColor(x.fidelity) }}>{fmt(x.fidelity, 1)}</span></div>
                <div className="small mt-8" style={{ lineHeight: 1.6 }}>{x.response || <span className="dim">(empty)</span>}</div>
                {x.failures.length > 0 && <div className="row wrap gap-4 mt-8">{x.failures.map(f => <Badge key={f} tone="red">{f}</Badge>)}</div>}
              </div>
            ))}
          </div>
        )) : <Empty icon="chat" title="No paired items" />}
      </Card>
    </>)}
  </>)
}

function ModelSide({ m, tone, label }) {
  return (
    <Card glow={tone === 'violet' ? 'v' : 'c'} className="pop-in">
      <div className="row between"><Badge tone={tone}>{label}</Badge><StatusBadge status={m.status} /></div>
      <div className="mono mt-8" style={{ fontSize: 18, fontWeight: 600 }}>{m.model}</div>
      <div className="tiny dim">{m.kind} · {compact(m.parameters)} params</div>
      <div className="mt-12" style={{ fontSize: 34, fontWeight: 750, letterSpacing: '-.03em', color: fidelityColor(m.fidelity) }}><CountUp value={m.fidelity} decimals={2} /></div>
      <div className="tiny dim">benchmark fidelity</div>
      <div className="divider" />
      <div className="col gap-4 small">
        <div className="row between"><span className="muted">Stability</span><span className="mono">{fmt(m.stability, 1)}</span></div>
        <div className="row between"><span className="muted">Consistency</span><span className="mono">{fmt(m.consistency, 1)}</span></div>
        <div className="row between"><span className="muted">Failure rate</span><span className="mono">{pct(m.failure_rate, 1)}</span></div>
        <div className="row between"><span className="muted">Val loss</span><span className="mono">{fmt(m.val_loss, 3)}</span></div>
        <div className="row between"><span className="muted">Avg length</span><span className="mono">{fmt(m.avg_length, 1)} words</span></div>
      </div>
    </Card>
  )
}
