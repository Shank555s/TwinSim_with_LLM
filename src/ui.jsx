import { createContext, useContext, useEffect, useRef, useState } from 'react'
import Icon from './icons.jsx'
import { STATUS_TONE, fidelityColor } from './api.js'

export const AppCtx = createContext(null)
export const useApp = () => useContext(AppCtx)

export function prefersReducedMotion() {
  try {
    return document.documentElement.classList.contains('reduce-motion') ||
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
  } catch { return false }
}

// Smoothly animates a number toward `value` (requestAnimationFrame; skipped under reduced motion).
export function useTween(value, duration = 900) {
  const [v, setV] = useState(value ?? 0)
  const from = useRef(value ?? 0)
  useEffect(() => {
    if (value === null || value === undefined || Number.isNaN(value)) return
    if (prefersReducedMotion()) { setV(value); from.current = value; return }
    const start = performance.now(), a = from.current, b = value
    let raf
    const step = now => {
      const t = Math.min(1, (now - start) / duration)
      const e = 1 - Math.pow(1 - t, 3)
      const cur = a + (b - a) * e
      setV(cur)
      from.current = cur
      if (t < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [value, duration])
  return v
}

export function CountUp({ value, decimals = 0, duration = 1000, prefix = '', suffix = '' }) {
  const v = useTween(value, duration)
  if (value === null || value === undefined) return <span>–</span>
  const s = decimals ? v.toFixed(decimals) : Math.round(v).toLocaleString('en-US')
  return <span className="num">{prefix}{s}{suffix}</span>
}

export function Card({ children, className = '', style, glow, hover, onClick, i }) {
  const cls = ['card', glow === 'v' ? 'glow-v' : glow === 'c' ? 'glow-c' : '', hover ? 'hover' : '', className].join(' ')
  return <div className={cls} style={i !== undefined ? { ...style, '--i': i } : style} onClick={onClick}>{children}</div>
}

export function CardHead({ title, sub, icon, iconTone = 'violet', right }) {
  return (
    <div className="card-head">
      <div className="grow">
        <h3 className="card-title">
          {icon && <span className={iconTone} style={{ display: 'inline-flex' }}><Icon name={icon} size={16} /></span>}
          {title}
        </h3>
        {sub && <div className="card-sub">{sub}</div>}
      </div>
      {right}
    </div>
  )
}

export function PageHead({ eyebrow, title, sub, right, tone }) {
  return (
    <div className="page-head">
      <div className="grow">
        {eyebrow && <div className={`eyebrow ${tone || ''}`}>{eyebrow}</div>}
        <h1 className="page-title">{title}</h1>
        {sub && <p className="page-sub">{sub}</p>}
      </div>
      {right && <div className="row wrap">{right}</div>}
    </div>
  )
}

export function Btn({ children, kind = '', size = '', icon, loading, disabled, onClick, title, type = 'button', className = '' }) {
  return (
    <button type={type} title={title} className={`btn ${kind} ${size} ${className}`} disabled={disabled || loading} onClick={onClick}>
      {loading ? <Icon name="replay" size={15} className="spin" /> : icon && <Icon name={icon} size={15} />}
      {children}
    </button>
  )
}

export function Badge({ children, tone = '', upper, title }) {
  return <span title={title} className={`badge ${tone} ${upper ? 'upper' : ''}`}>{children}</span>
}

export function StatusBadge({ status }) {
  const tone = STATUS_TONE[status] ?? ''
  return <Badge tone={status === 'active' ? 'solid-v' : tone} upper>{status === 'active' && <span className="dot pulse" style={{ background: '#fff', width: 6, height: 6 }} />}{status}</Badge>
}

export function Dot({ tone = '', pulse }) { return <span className={`dot ${tone} ${pulse ? 'pulse' : ''}`} /> }

export function Empty({ icon = 'sparkle', title, children, action }) {
  return (
    <div className="empty fade-in">
      <div className="empty-icon"><Icon name={icon} size={28} /></div>
      <h4>{title}</h4>
      {children && <p>{children}</p>}
      {action && <div className="mt-8">{action}</div>}
    </div>
  )
}

export function Thinking({ children }) {
  return <div className="thinking"><span className="orb" /><span>{children}</span></div>
}

export function ErrorBox({ error, children }) {
  if (!error && !children) return null
  return (
    <div className="err-box fade-in">
      <Icon name="xCircle" size={17} style={{ flex: 'none', marginTop: 1 }} />
      <div>{children || error?.message || String(error)}</div>
    </div>
  )
}

export function Note({ tone = 'info', children, icon }) {
  return (
    <div className={`${tone}-box`}>
      <Icon name={icon || (tone === 'warn' ? 'warn' : 'info')} size={16} style={{ flex: 'none', marginTop: 2 }} />
      <div>{children}</div>
    </div>
  )
}

export function Skeleton({ h = 16, w = '100%', style }) { return <div className="skel" style={{ height: h, width: w, ...style }} /> }

export function Loading({ rows = 3 }) {
  return <div className="col">{Array.from({ length: rows }, (_, i) => <Skeleton key={i} h={i === 0 ? 22 : 14} w={i === 0 ? '40%' : `${90 - i * 12}%`} />)}</div>
}

export function Metric({ label, icon, tone = 'violet', value, decimals = 0, suffix, foot, i }) {
  return (
    <Card className={`metric ${tone === 'cyan' ? 'c' : ''}`} hover i={i}>
      <div className="m-label"><span className={`m-icon ${tone}`}><Icon name={icon} size={15} /></span>{label}</div>
      <div className="m-value">
        {typeof value === 'number' ? <CountUp value={value} decimals={decimals} /> : value ?? '–'}
        {suffix && value !== null && value !== undefined && <small>{suffix}</small>}
      </div>
      {foot && <div className="m-foot">{foot}</div>}
    </Card>
  )
}

export function Seg({ value, options, onChange }) {
  return (
    <div className="seg" role="tablist">
      {options.map(o => {
        const [v, l] = Array.isArray(o) ? o : [o, o]
        return <button key={v} role="tab" aria-selected={value === v} className={value === v ? 'on' : ''} onClick={() => onChange(v)}>{l}</button>
      })}
    </div>
  )
}

export function Toggle({ on, onChange, label }) {
  return (
    <label className="row gap-6" style={{ cursor: 'pointer' }}>
      <button type="button" className={`toggle ${on ? 'on' : ''}`} aria-pressed={on} onClick={() => onChange(!on)} />
      {label && <span className="small muted">{label}</span>}
    </label>
  )
}

export function Slider({ label, value, onChange, min = 0, max = 100, step = 1, color = 'var(--accent)', format }) {
  const p = ((value - min) / (max - min)) * 100
  return (
    <div className="slider-row">
      <span className="s-name">{label}</span>
      <input type="range" className="slider" min={min} max={max} step={step} value={value}
        style={{ '--p': `${p}%`, '--c': color }} onChange={e => onChange(Number(e.target.value))} aria-label={label} />
      <span className="s-val">{format ? format(value) : value}</span>
    </div>
  )
}

export function Progress({ value, indeterminate }) {
  return <div className={`progress ${indeterminate ? 'indet' : ''}`}><i style={{ width: `${Math.max(0, Math.min(1, value || 0)) * 100}%` }} /></div>
}

// Circular fidelity gauge (0-100). The arc animates from 0 to the measured value.
export function Gauge({ value, size = 180, label = 'Fidelity', sub }) {
  const r = (size - 18) / 2, c = 2 * Math.PI * r
  const [shown, setShown] = useState(0)
  useEffect(() => { const t = requestAnimationFrame(() => setShown(value ?? 0)); return () => cancelAnimationFrame(t) }, [value])
  const col = fidelityColor(value)
  return (
    <div className="gauge" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(237,232,226,.1)" strokeWidth="10" />
        {Array.from({ length: 40 }, (_, k) => {
          const a = (k / 40) * 2 * Math.PI
          const x1 = size / 2 + Math.cos(a) * (r - 12), y1 = size / 2 + Math.sin(a) * (r - 12)
          const x2 = size / 2 + Math.cos(a) * (r - 16), y2 = size / 2 + Math.sin(a) * (r - 16)
          return <line key={k} x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(237,232,226,.18)" strokeWidth="1" />
        })}
        <circle className="arc" cx={size / 2} cy={size / 2} r={r} fill="none" stroke={col} strokeWidth="10" strokeLinecap="butt"
          strokeDasharray={c} strokeDashoffset={c * (1 - shown / 100)} />
      </svg>
      <div className="g-val">
        <div className="g-num" style={{ color: col }}><CountUp value={value} decimals={1} duration={1300} /></div>
        <div className="g-cap">{label}</div>
        {sub && <div className="tiny dim mt-4">{sub}</div>}
      </div>
    </div>
  )
}

export function Modal({ open, onClose, children, width }) {
  useEffect(() => {
    if (!open) return
    const k = e => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [open, onClose])
  if (!open) return null
  return (<>
    <div className="overlay" onClick={onClose} />
    <div className="modal" role="dialog" style={width ? { width: `min(${width}px, calc(100vw - 32px))` } : undefined}>{children}</div>
  </>)
}

export function Drawer({ open, onClose, children }) {
  useEffect(() => {
    if (!open) return
    const k = e => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [open, onClose])
  if (!open) return null
  return (<>
    <div className="overlay" onClick={onClose} />
    <aside className="drawer" role="dialog">{children}</aside>
  </>)
}

export function copyText(text, toast) {
  const done = () => toast?.('Copied to clipboard', 'ok')
  try {
    navigator.clipboard.writeText(text).then(done, () => fallback())
  } catch { fallback() }
  function fallback() {
    const ta = document.createElement('textarea')
    ta.value = text
    document.body.appendChild(ta)
    ta.select()
    try { document.execCommand('copy'); done() } catch { /* ignore */ }
    ta.remove()
  }
}

export function Field({ label, children, hint }) {
  return (
    <div className="field">
      {label && <label>{label}</label>}
      {children}
      {hint && <span className="tiny dim">{hint}</span>}
    </div>
  )
}
