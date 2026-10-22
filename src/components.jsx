// Domain components shared by several pages. Everything renders backend data as-is.
import { useState } from 'react'
import Icon from './icons.jsx'
import { LineChart } from './charts.jsx'
import { AXIS_LABEL, compact, dateTime, fmt, pct, shortModel, signed } from './api.js'
import { Badge, Btn, StatusBadge, useApp } from './ui.jsx'

const STATUS_COLOR = { baseline: '#7FA7B8', active: '#E8772E', archived: '#D99A6C', rejected: '#D96459', candidate: '#D9A441' }
export const statusColor = s => STATUS_COLOR[s] || '#7A7168'
export const isAccepted = m => ['baseline', 'active', 'archived'].includes(m.status)

export function PairedLabel({ p, short }) {
  if (!p) return <span className="dim">no paired data</span>
  const sig = Math.abs(p.z) >= 2
  return (
    <span className={`mono ${p.mean >= 0 ? (sig ? 'green' : 'muted') : (sig ? 'red' : 'muted')}`} title={`paired over ${p.n} identical benchmark items`}>
      {signed(p.mean, 2)} ± {fmt(p.se, 2)}{short ? '' : ` · z ${fmt(p.z, 2)}`}{!short && !sig ? ' (noise)' : ''}
    </span>
  )
}

// Behavioral fidelity across every model version: accepted lineage as a line, rejected candidates as hollow red points.
export function FidelityCurve({ curve, height = 270, compact: small }) {
  const acc = curve.filter(isAccepted)
  const rej = curve.filter(m => !isAccepted(m))
  const base = curve.find(m => m.status === 'baseline')
  const toPt = m => ({ x: m.version, y: m.fidelity, meta: m, tone: statusColor(m.status), hollow: m.status === 'rejected', label: small ? undefined : `v${m.version}` })
  const tooltip = p => {
    const m = p.meta
    return (<>
      <div className="tip-title"><span className="mono">{m.model}</span><StatusBadge status={m.status} /></div>
      <div className="row between"><span className="dim">Fidelity</span><b className="mono">{fmt(m.fidelity, 2)}</b></div>
      <div className="row between"><span className="dim">Update</span><span className="mono">{m.kind}</span></div>
      {m.paired_vs_parent && <div className="row between gap-16"><span className="dim">vs {shortModel(m.parent)}</span><PairedLabel p={m.paired_vs_parent} short /></div>}
      <div className="row between"><span className="dim">Val loss</span><span className="mono">{fmt(m.val_loss, 3)}</span></div>
      <div className="row between"><span className="dim">Failure rate</span><span className="mono">{pct(m.failure_rate, 1)}</span></div>
    </>)
  }
  return (
    <LineChart height={height} area pulseLast tooltip={tooltip} minXStep={1}
      xLabel={v => (Number.isInteger(v) ? `v${v}` : '')} yLabel={v => v.toFixed(0)}
      refLine={base && curve.length > 1 ? { y: base.fidelity, label: `baseline ${fmt(base.fidelity, 1)}`, color: '#7FA7B8' } : null}
      series={[
        { key: 'acc', label: 'Accepted lineage', color: '#E8772E', points: acc.map(toPt) },
        { key: 'rej', label: 'Rejected candidates', color: '#D96459', noLine: true, points: rej.map(toPt) },
      ]} />
  )
}

export function EvolutionTimeline({ curve, selected, onSelect }) {
  return (
    <div className="evo">
      {curve.map((m, i) => (
        <button key={m.model} className={`evo-node ${m.status} ${selected === m.model ? 'sel' : ''}`} style={{ '--i': i }} onClick={() => onSelect(selected === m.model ? null : m.model)}>
          {i < curve.length - 1 && <span className={`evo-link ${curve[i + 1].status === 'rejected' ? 'from-rejected' : ''}`} style={{ '--i': i }} />}
          <span className="core">v{m.version}</span>
          <span className="nm">{m.model}</span>
          <span className="sc">{fmt(m.fidelity, 1)}</span>
          <StatusBadge status={m.status} />
        </button>
      ))}
    </div>
  )
}

export function ModelDetail({ m, onClose }) {
  const { nav } = useApp()
  if (!m) return null
  const reasons = m.decision?.reasons || []
  return (
    <div className="card elev pop-in mt-12" style={{ borderColor: `${statusColor(m.status)}55` }}>
      <div className="row between wrap">
        <div className="row"><span className="mono" style={{ fontSize: 16, fontWeight: 600 }}>{m.model}</span><StatusBadge status={m.status} /><Badge>{m.kind}</Badge></div>
        <div className="row">
          <Btn size="sm" kind="ghost" icon="dna" onClick={() => nav('dna', { model: m.model })}>Twin DNA</Btn>
          <Btn size="sm" icon="layers" onClick={() => nav('registry', { model: m.model })}>Open in registry</Btn>
          {onClose && <button className="icon-btn" onClick={onClose} aria-label="Close"><Icon name="x" size={14} /></button>}
        </div>
      </div>
      <div className="vmetrics">
        <div className="vmetric"><div className="k">Fidelity</div><div className="v">{fmt(m.fidelity, 2)}</div></div>
        <div className="vmetric"><div className="k">vs parent</div><div className="v">{m.paired_vs_parent ? <PairedLabel p={m.paired_vs_parent} /> : '–'}</div></div>
        <div className="vmetric"><div className="k">Stability</div><div className="v">{fmt(m.stability, 1)}</div></div>
        <div className="vmetric"><div className="k">Consistency</div><div className="v">{fmt(m.consistency, 1)}</div></div>
        <div className="vmetric"><div className="k">Failure rate</div><div className="v">{pct(m.failure_rate, 1)}</div></div>
        <div className="vmetric"><div className="k">Val loss</div><div className="v">{fmt(m.val_loss, 3)}</div></div>
        <div className="vmetric"><div className="k">Params</div><div className="v">{compact(m.parameters)}</div></div>
        <div className="vmetric"><div className="k">Parent</div><div className="v">{m.parent || '–'}</div></div>
        <div className="vmetric"><div className="k">Created</div><div className="v">{dateTime(m.created_at)}</div></div>
      </div>
      {reasons.length > 0 && (
        <div className="mt-12">
          <div className="section-label">Acceptance decision</div>
          <ul className="list-plain small muted">{reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
        </div>
      )}
    </div>
  )
}

export function PersonaPicker({ personas, value, onChange }) {
  return (
    <div className="persona-pick">
      {personas.map(p => (
        <button key={p.id} className={`persona-btn ${value === p.id ? 'on' : ''}`} onClick={() => onChange(p.id)}>
          <div className="pb-name ellipsis">{p.label || p.id}</div>
          <div className="tiny dim ellipsis">{[p.age, p.occupation].filter(Boolean).join(' · ')}</div>
          <div className="row gap-4 mt-4">{p.is_benchmark ? <Badge tone="cyan">benchmark</Badge> : <Badge tone="violet">custom</Badge>}</div>
        </button>
      ))}
    </div>
  )
}

export function AxisTable({ axes }) {
  return (
    <div className="col gap-4">
      {Object.entries(axes || {}).map(([k, v]) => (
        <div key={k} className="row between small"><span className="muted">{AXIS_LABEL[k] || k}</span><span className="mono">{fmt(v, 2)}</span></div>
      ))}
    </div>
  )
}

export function Collapse({ title, children, defaultOpen }) {
  const [open, setOpen] = useState(!!defaultOpen)
  return (
    <div>
      <button className="btn ghost sm" onClick={() => setOpen(!open)} style={{ paddingLeft: 0 }}>
        <Icon name="chevronR" size={14} style={{ transform: open ? 'rotate(90deg)' : 'none', transition: 'transform .2s' }} />{title}
      </button>
      {open && <div className="fade-in mt-8">{children}</div>}
    </div>
  )
}

