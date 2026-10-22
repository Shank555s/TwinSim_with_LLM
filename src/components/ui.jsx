import { FAILURE_HELP, FAILURE_LABELS } from '../constants.js'

export function Page({ title, intro, actions, children }) {
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>{title}</h1>
          {intro && <p>{intro}</p>}
        </div>
        {actions && <div className="row">{actions}</div>}
      </header>
      <div className="stack" style={{ gap: 16 }}>{children}</div>
    </div>
  )
}

export function Panel({ title, hint, actions, children, style, className = '' }) {
  return (
    <section className={`panel ${className}`} style={style}>
      {(title || actions) && (
        <div className="panel-head">
          <div className="row" style={{ gap: 10 }}>
            {title && <h2>{title}</h2>}
            {hint && <span className="hint">{hint}</span>}
          </div>
          {actions && <div className="row">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  )
}

export function Figures({ items }) {
  return (
    <div className="figures">
      {items.map((x) => (
        <div className="figure" key={x.l}>
          <div className="v" style={x.color ? { color: x.color } : undefined}>{x.v}</div>
          <div className="l">{x.l}</div>
          {x.d && <div className="d" style={{ color: x.dColor || 'var(--faint)' }}>{x.d}</div>}
        </div>
      ))}
    </div>
  )
}

export function Btn({ kind = '', small, children, ...rest }) {
  return <button className={`btn ${kind} ${small ? 'small' : ''}`} {...rest}>{children}</button>
}

export function Seg({ value, options, onChange, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.v} type="button" aria-pressed={value === o.v} onClick={() => onChange(o.v)}>{o.l}</button>
      ))}
    </div>
  )
}

export function Field({ label, children, hint }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <span className="faint">{hint}</span>}
    </label>
  )
}

export function Slider({ label, value, onChange, lo, hi, min = 0, max = 100, step = 1 }) {
  return (
    <div className="slider">
      <span>{label}</span>
      <output>{value}</output>
      <input type="range" min={min} max={max} step={step} value={value} aria-label={label}
        onChange={(e) => onChange(Number(e.target.value))} />
      {(lo || hi) && <div className="ends"><span>{lo}</span><span>{hi}</span></div>}
    </div>
  )
}

export function Tag({ kind = '', children, title }) {
  return <span className={`tag ${kind}`} title={title}>{children}</span>
}

export function FailureTags({ codes, empty = 'No failure modes' }) {
  if (!codes || !codes.length) return <Tag kind="good">{empty}</Tag>
  return (
    <span className="row" style={{ gap: 6 }}>
      {codes.map((c) => <Tag kind="bad" key={c} title={FAILURE_HELP[c]}>{FAILURE_LABELS[c] || c}</Tag>)}
    </span>
  )
}

export function StatusTag({ status }) {
  const kind = { active: 'good', baseline: 'persona', rejected: 'bad', candidate: 'warn', archived: '' }[status] ?? ''
  return <Tag kind={kind}>{status}</Tag>
}

export function ErrorNote({ error, onRetry }) {
  if (!error) return null
  return (
    <div className="banner bad" role="alert">
      <div style={{ flex: 1 }}>
        <strong>{error.code === 'model_not_trained' ? 'No trained model yet' : 'Something went wrong'}</strong>
        <p className="sub" style={{ marginTop: 2 }}>{error.message}</p>
      </div>
      {onRetry && <Btn small onClick={onRetry}>Try again</Btn>}
    </div>
  )
}

export function Empty({ children, action }) {
  return <div className="empty-state"><div>{children}</div>{action}</div>
}

export function Progress({ value }) {
  return <div className="bar" role="progressbar" aria-valuenow={Math.round((value || 0) * 100)} aria-valuemin={0} aria-valuemax={100}><i style={{ width: `${Math.max(0, Math.min(1, value || 0)) * 100}%` }} /></div>
}

export function Loading({ what = 'Loading' }) {
  return <p className="faint">{what}…</p>
}
