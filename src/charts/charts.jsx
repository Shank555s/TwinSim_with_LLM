// Small dependency-free SVG charts. Colour meaning is fixed app-wide:
// persona / intended = sand, twin / observed / adaptive = periwinkle, failures = coral.
import { useEffect, useRef, useState } from 'react'

// Render charts at their real pixel width so text stays 11px at any panel size.
function useWidth(fallback = 640) {
  const ref = useRef(null)
  const [w, setW] = useState(fallback)
  useEffect(() => {
    if (!ref.current || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(([e]) => setW(Math.max(200, Math.round(e.contentRect.width))))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, w]
}

const C = { line: '#2b3543', text: '#9aa4b2', faint: '#6b7686', persona: '#d9c9a3', twin: '#93a4ff', bad: '#ee7c6b', good: '#7fc59b' }

function niceTicks(min, max, n = 4, minStep = 0) {
  if (min === max) { min -= 1; max += 1 }
  const step0 = Math.max(minStep, (max - min) / n)
  const mag = 10 ** Math.floor(Math.log10(step0))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0
  const lo = Math.floor(min / step) * step
  const hi = Math.ceil(max / step) * step
  const out = []
  for (let v = lo; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6))
  return out
}

/** Line chart. series: [{name, color, dashed, points:[{x(label), y}]}] sharing x labels. */
export function LineChart({ labels, series, height = 220, yLabel, yDomain, format = (v) => v.toFixed(1), markers, minStep = 0 }) {
  const [hover, setHover] = useState(null)
  const [ref, W] = useWidth()
  const H = height, L = 46, R = 14, T = 12, B = 30
  const all = series.flatMap((s) => s.points.map((p) => p.y)).filter((v) => v !== null && v !== undefined)
  if (!labels.length || !all.length) return <div ref={ref}><p className="faint">No data yet.</p></div>
  const ticks = niceTicks(yDomain ? yDomain[0] : Math.min(...all), yDomain ? yDomain[1] : Math.max(...all), 4, minStep)
  const y0 = ticks[0], y1 = ticks[ticks.length - 1]
  const xs = (i) => L + (labels.length === 1 ? (W - L - R) / 2 : (i * (W - L - R)) / (labels.length - 1))
  const ys = (v) => T + (1 - (v - y0) / (y1 - y0 || 1)) * (H - T - B)
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ display: 'block' }} role="img" aria-label={yLabel || 'line chart'}
        onMouseLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={ys(t)} y2={ys(t)} stroke={C.line} />
            <text x={L - 8} y={ys(t) + 4} fill={C.faint} fontSize="11" textAnchor="end">{format(t)}</text>
          </g>
        ))}
        {labels.map((lb, i) => (
          <text key={lb + i} x={xs(i)} y={H - 10} fill={C.text} fontSize="11" textAnchor="middle">{lb}</text>
        ))}
        {series.map((s) => {
          const pts = s.points.map((p, i) => (p.y === null || p.y === undefined ? null : [xs(i), ys(p.y), p, i])).filter(Boolean)
          return (
            <g key={s.name}>
              <polyline fill="none" stroke={s.color} strokeWidth="2" strokeDasharray={s.dashed ? '5 4' : undefined}
                points={pts.map((p) => `${p[0]},${p[1]}`).join(' ')} />
              {pts.map(([x, y, p, i]) => (
                <circle key={i} cx={x} cy={y} r={hover === i ? 5 : 3.5} fill={p.hollow ? '#151b23' : s.color} stroke={s.color} strokeWidth="2" />
              ))}
            </g>
          )
        })}
        {labels.map((lb, i) => (
          <rect key={'h' + i} x={xs(i) - 14} y={T} width={28} height={H - T - B} fill="transparent"
            onMouseEnter={() => setHover(i)} />
        ))}
        {hover !== null && <line x1={xs(hover)} x2={xs(hover)} y1={T} y2={H - B} stroke={C.faint} strokeDasharray="2 3" />}
      </svg>
      {hover !== null && (
        <div style={{ position: 'absolute', top: 4, left: `${(xs(hover) / W) * 100}%`, transform: 'translateX(-50%)', background: '#222c3a', border: '1px solid #3a4657', borderRadius: 6, padding: '6px 10px', fontSize: 12, pointerEvents: 'none', whiteSpace: 'nowrap' }}>
          <div style={{ color: C.text }}>{labels[hover]}</div>
          {series.map((s) => s.points[hover]?.y != null && (
            <div key={s.name} style={{ color: s.color }}>{s.name}: {format(s.points[hover].y)}{s.points[hover].note ? ` (${s.points[hover].note})` : ''}</div>
          ))}
        </div>
      )}
      {markers}
    </div>
  )
}

/** Horizontal bars: rows [{label, value, color?, note?}] with value in [0, max]. */
export function HBars({ rows, max, format = (v) => v.toFixed(1), color = C.twin }) {
  const m = max ?? Math.max(1e-9, ...rows.map((r) => r.value || 0))
  return (
    <div className="stack" style={{ gap: 8 }}>
      {rows.map((r) => (
        <div key={r.label} style={{ display: 'grid', gridTemplateColumns: 'minmax(110px, 180px) 1fr 56px', gap: 10, alignItems: 'center', fontSize: 13 }}>
          <span className="sub" title={r.title}>{r.label}</span>
          <div style={{ background: '#222c3a', borderRadius: 2, height: 10 }}>
            <div style={{ width: `${Math.min(100, ((r.value || 0) / m) * 100)}%`, height: '100%', borderRadius: 2, background: r.color || color }} />
          </div>
          <span style={{ textAlign: 'right' }}>{r.value === null || r.value === undefined ? '—' : format(r.value)}</span>
        </div>
      ))}
    </div>
  )
}

/** Diverging bars for drift (observed - intended), domain [-span, span]. */
export function DriftBars({ rows, span = 5 }) {
  return (
    <div className="stack" style={{ gap: 7 }}>
      {rows.map((r) => {
        const v = r.value ?? 0
        const w = Math.min(50, (Math.abs(v) / span) * 50)
        return (
          <div key={r.label} style={{ display: 'grid', gridTemplateColumns: 'minmax(110px, 170px) 1fr 54px', gap: 10, alignItems: 'center', fontSize: 13 }}>
            <span className="sub">{r.label}</span>
            <div style={{ position: 'relative', height: 12, background: '#1c2430' }}>
              <div style={{ position: 'absolute', left: '50%', top: -3, bottom: -3, width: 1, background: '#6b7686' }} />
              <div style={{ position: 'absolute', top: 1, bottom: 1, borderRadius: 2, background: Math.abs(v) < 1 ? C.good : C.bad, left: v < 0 ? `${50 - w}%` : '50%', width: `${w}%` }} />
            </div>
            <span style={{ textAlign: 'right', color: Math.abs(v) < 1 ? C.good : C.bad }}>{v > 0 ? '+' : ''}{v.toFixed(1)}</span>
          </div>
        )
      })}
    </div>
  )
}

/** Radar with two polygons (intended vs observed or axis scores). values 0..10 */
export function Radar({ axes, a, b, aName = 'Intended', bName = 'Twin', size = 300 }) {
  const n = axes.length, cx = size / 2, cy = size / 2, r = size / 2 - 58
  const pt = (i, v) => {
    const ang = -Math.PI / 2 + (i * 2 * Math.PI) / n
    return [cx + Math.cos(ang) * r * (v / 10), cy + Math.sin(ang) * r * (v / 10)]
  }
  const poly = (vals) => vals.map((v, i) => pt(i, v ?? 0).join(',')).join(' ')
  return (
    <svg viewBox={`0 0 ${size} ${size}`} width="100%" style={{ maxWidth: size }} role="img" aria-label="radar chart">
      {[2.5, 5, 7.5, 10].map((g) => <polygon key={g} points={poly(axes.map(() => g))} fill="none" stroke={C.line} />)}
      {axes.map((ax, i) => {
        const [x, y] = pt(i, 10)
        const [lx, ly] = pt(i, 12.4)
        return (
          <g key={ax.k}>
            <line x1={cx} y1={cy} x2={x} y2={y} stroke={C.line} />
            <text x={lx} y={ly + 4} fill={C.text} fontSize="10.5" textAnchor="middle">{ax.short || ax.l}</text>
          </g>
        )
      })}
      {a && <polygon points={poly(a)} fill="rgba(217,201,163,.12)" stroke={C.persona} strokeWidth="1.5" strokeDasharray="4 3" />}
      {b && <polygon points={poly(b)} fill="rgba(147,164,255,.2)" stroke={C.twin} strokeWidth="2" />}
      <g fontSize="11">
        {a && <text x={8} y={size - 8} fill={C.persona}>— — {aName}</text>}
        {b && <text x={size - 8} y={size - 8} fill={C.twin} textAnchor="end">—— {bName}</text>}
      </g>
    </svg>
  )
}

/** Specimen strip: 20 cells, value 0..10 */
export function Cells({ value, kind }) {
  const on = Math.round(((value ?? 0) / 10) * 20)
  return (
    <div className={`cells ${kind}`} aria-hidden="true">
      {Array.from({ length: 20 }, (_, i) => <i key={i} className={i < on ? 'on' : ''} />)}
    </div>
  )
}

/** Small multi-row "trend" bars per model version (failure-mode trend). */
export function VersionBars({ rows, max = 1, format = (v) => `${(v * 100).toFixed(0)}%`, color = C.bad }) {
  return (
    <div className="stack" style={{ gap: 4 }}>
      {rows.map((r) => (
        <div key={r.label} style={{ display: 'grid', gridTemplateColumns: '92px 1fr 44px', gap: 8, alignItems: 'center', fontSize: 12.5 }}>
          <span className="faint" style={{ color: r.highlight ? '#e6e8ec' : undefined }}>{r.label}</span>
          <div style={{ height: 8, background: '#222c3a', borderRadius: 2 }}>
            {r.value != null && <div style={{ width: `${Math.min(100, (r.value / max) * 100)}%`, height: '100%', background: r.dim ? '#6b7686' : color, borderRadius: 2 }} />}
          </div>
          <span style={{ textAlign: 'right' }}>{r.value == null ? '—' : format(r.value)}</span>
        </div>
      ))}
    </div>
  )
}

/** Loss curve from training-run log events. */
export function LossChart({ log, height = 200 }) {
  const [ref, W] = useWidth()
  const train = (log || []).filter((e) => e.type === 'train' || e.kind === 'train')
  const evals = (log || []).filter((e) => e.type === 'eval' || e.kind === 'eval')
  if (train.length < 2) return <div ref={ref}><p className="faint">Waiting for training steps…</p></div>
  const H = height, L = 46, R = 14, T = 10, B = 26
  const maxStep = Math.max(...train.map((e) => e.step), ...evals.map((e) => e.step))
  const vals = [...train.map((e) => e.train_loss_ema ?? e.train_loss), ...evals.map((e) => e.val_loss)]
  const ticks = niceTicks(Math.min(...vals), Math.max(...vals))
  const y0 = ticks[0], y1 = ticks[ticks.length - 1]
  const xs = (s) => L + (s / (maxStep || 1)) * (W - L - R)
  const ys = (v) => T + (1 - (v - y0) / (y1 - y0 || 1)) * (H - T - B)
  return (
    <div ref={ref}><svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ display: 'block' }} role="img" aria-label="loss curve">
      {ticks.map((t) => (
        <g key={t}><line x1={L} x2={W - R} y1={ys(t)} y2={ys(t)} stroke={C.line} />
          <text x={L - 8} y={ys(t) + 4} fill={C.faint} fontSize="11" textAnchor="end">{t.toFixed(2)}</text></g>
      ))}
      <text x={W - R} y={H - 6} fill={C.faint} fontSize="11" textAnchor="end">step {maxStep}</text>
      <polyline fill="none" stroke={C.twin} strokeWidth="1.8" points={train.map((e) => `${xs(e.step)},${ys(e.train_loss_ema ?? e.train_loss)}`).join(' ')} />
      <polyline fill="none" stroke={C.persona} strokeWidth="1.8" strokeDasharray="4 3" points={evals.map((e) => `${xs(e.step)},${ys(e.val_loss)}`).join(' ')} />
      {evals.map((e, i) => <circle key={i} cx={xs(e.step)} cy={ys(e.val_loss)} r="3" fill={C.persona} />)}
    </svg></div>
  )
}
