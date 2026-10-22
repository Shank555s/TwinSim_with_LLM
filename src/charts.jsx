// Lightweight SVG charts (no chart library). Rendered at the measured container width so text never scales.
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { AXIS_LABEL, fidelityColor } from './api.js'

export function useWidth(min = 200) {
  const ref = useRef(null)
  const [w, setW] = useState(0)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    setW(el.clientWidth)
    const ro = new ResizeObserver(([e]) => setW(Math.max(min, Math.floor(e.contentRect.width))))
    ro.observe(el)
    return () => ro.disconnect()
  }, [min])
  return [ref, w]
}

// Monotone cubic interpolation (Fritsch-Carlson): smooth without overshooting the data.
export function smoothPath(pts) {
  const n = pts.length
  if (n === 0) return ''
  if (n === 1) return `M${pts[0][0]},${pts[0][1]}`
  if (n === 2) return `M${pts[0][0]},${pts[0][1]}L${pts[1][0]},${pts[1][1]}`
  const dx = [], dy = [], m = [], t = []
  for (let i = 0; i < n - 1; i++) { dx[i] = pts[i + 1][0] - pts[i][0]; dy[i] = pts[i + 1][1] - pts[i][1]; m[i] = dy[i] / (dx[i] || 1e-9) }
  t[0] = m[0]; t[n - 1] = m[n - 2]
  for (let i = 1; i < n - 1; i++) t[i] = m[i - 1] * m[i] <= 0 ? 0 : (m[i - 1] + m[i]) / 2
  for (let i = 0; i < n - 1; i++) {
    if (m[i] === 0) { t[i] = 0; t[i + 1] = 0; continue }
    const a = t[i] / m[i], b = t[i + 1] / m[i], s = a * a + b * b
    if (s > 9) { const k = 3 / Math.sqrt(s); t[i] = k * a * m[i]; t[i + 1] = k * b * m[i] }
  }
  let d = `M${pts[0][0]},${pts[0][1]}`
  for (let i = 0; i < n - 1; i++) {
    const h = dx[i] / 3
    d += `C${pts[i][0] + h},${pts[i][1] + t[i] * h} ${pts[i + 1][0] - h},${pts[i + 1][1] - t[i + 1] * h} ${pts[i + 1][0]},${pts[i + 1][1]}`
  }
  return d
}

function niceTicks(lo, hi, count = 4) {
  const span = hi - lo || 1
  const raw = span / count
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => s >= raw) || raw
  const out = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6))
  return out
}

function PathLen({ d, className, stroke, dash, width = 2.4, opacity = 1 }) {
  const ref = useRef(null)
  const [len, setLen] = useState(3000)
  useEffect(() => { try { setLen(Math.ceil(ref.current.getTotalLength()) + 2) } catch { /* jsdom */ } }, [d])
  return <path ref={ref} d={d} className={`line-path ${dash ? '' : className}`} stroke={stroke} strokeWidth={width} opacity={opacity}
    strokeDasharray={dash || undefined} style={dash ? undefined : { '--len': len }} />
}

/**
 * series: [{ key, label, color, dashed, points: [{ x, y, meta, hollow, tone }] }]
 * x is numeric (index or step). xLabel(x) formats ticks; tooltip(point, series) renders rich content.
 */
export function LineChart({ series, height = 260, yDomain, xTicks, xLabel = v => v, yLabel = v => v, tooltip,
  pulseLast, area, minXStep = 1, yTitle, refLine }) {
  const [ref, w] = useWidth()
  const [hover, setHover] = useState(null)
  const pad = { l: 44, r: 18, t: 16, b: 30 }
  const all = series.flatMap(s => s.points).filter(p => p.y !== null && p.y !== undefined)
  const xs = all.map(p => p.x)
  const xMin = Math.min(...xs, 0), xMax = Math.max(...xs, xMin + minXStep)
  let [yMin, yMax] = yDomain || [Math.min(...all.map(p => p.y)), Math.max(...all.map(p => p.y))]
  if (!yDomain) { const padY = (yMax - yMin) * 0.15 || 1; yMin -= padY; yMax += padY }
  const W = Math.max(w, 200), H = height
  const X = x => pad.l + ((x - xMin) / (xMax - xMin || 1)) * (W - pad.l - pad.r)
  const Y = y => pad.t + (1 - (y - yMin) / (yMax - yMin || 1)) * (H - pad.t - pad.b)
  const yt = niceTicks(yMin, yMax, 4).filter(v => v >= yMin && v <= yMax)
  const xt = useMemo(() => {
    if (xTicks) return xTicks
    const uniq = [...new Set(xs)].sort((a, b) => a - b)
    const maxTicks = Math.max(2, Math.floor((W - pad.l - pad.r) / 70))
    if (uniq.length <= maxTicks) return uniq
    return niceTicks(xMin, xMax, Math.min(6, maxTicks)).filter(v => v >= xMin && v <= xMax)
  }, [xTicks, xs.join(','), W]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="chart" ref={ref} onMouseLeave={() => setHover(null)}>
      {w > 0 && all.length > 0 && (
        <svg width={W} height={H}>
          <g className="axis">
            {yt.map(v => (
              <g key={v}>
                <line className="gridline" x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} />
                <text x={pad.l - 8} y={Y(v) + 3.5} textAnchor="end">{yLabel(v)}</text>
              </g>
            ))}
            {xt.map(v => <text key={v} x={X(v)} y={H - 8} textAnchor="middle">{xLabel(v)}</text>)}
            {yTitle && <text x={pad.l} y={9} textAnchor="start" style={{ letterSpacing: '.08em' }}>{yTitle}</text>}
          </g>
          {refLine && (
            <g>
              <line x1={pad.l} x2={W - pad.r} y1={Y(refLine.y)} y2={Y(refLine.y)} stroke={refLine.color || '#7FA7B8'} strokeDasharray="4 5" opacity=".55" />
              <text x={W - pad.r} y={Y(refLine.y) - 6} textAnchor="end" fill={refLine.color || '#7FA7B8'} fontSize="10.5" fontFamily="var(--mono)">{refLine.label}</text>
            </g>
          )}
          {series.map((s, si) => {
            const pts = s.points.filter(p => p.y !== null && p.y !== undefined).map(p => [X(p.x), Y(p.y)])
            if (!pts.length || s.noLine) return null
            const d = smoothPath(pts)
            return (
              <g key={s.key || si}>
                {area && si === 0 && pts.length > 1 && (
                  <path className="area" d={`${d}L${pts[pts.length - 1][0]},${H - pad.b}L${pts[0][0]},${H - pad.b}Z`} fill={s.color} fillOpacity=".08" />
                )}
                <PathLen d={d} className="draw" stroke={s.color} dash={s.dashed ? '5 5' : null} width={s.width || 2.4} opacity={s.opacity ?? 1} />
              </g>
            )
          })}
          {series.map((s, si) => !s.noPoints && s.points.filter(p => p.y !== null && p.y !== undefined).map((p, pi, arr) => {
            const last = pulseLast && si === 0 && pi === arr.length - 1
            const col = p.tone || s.color
            const on = hover && hover.si === si && hover.pi === pi
            return (
              <g key={`${si}-${pi}`} onMouseEnter={() => setHover({ si, pi, p, s, x: X(p.x), y: Y(p.y) })}>
                {last && <circle className="pulse-ring" cx={X(p.x)} cy={Y(p.y)} r="7" fill="none" stroke={col} strokeWidth="2" />}
                <circle cx={X(p.x)} cy={Y(p.y)} r="14" fill="transparent" />
                <circle className="pt pt-in" style={{ animationDelay: `${300 + pi * 80}ms` }} cx={X(p.x)} cy={Y(p.y)} r={on ? 6.5 : last ? 5.5 : 4.2}
                  fill={p.hollow ? '#12100E' : col} stroke={col} strokeWidth={p.hollow ? 2 : 1.5} />
                {p.label && <text x={X(p.x)} y={Y(p.y) - 12} textAnchor="middle" fontSize="10.5" fill="#A8A097" fontFamily="var(--mono)">{p.label}</text>}
              </g>
            )
          }))}
          {hover && <line x1={hover.x} x2={hover.x} y1={pad.t} y2={H - pad.b} stroke="rgba(217,154,108,.25)" strokeDasharray="3 4" pointerEvents="none" />}
        </svg>
      )}
      {hover && tooltip && (
        <div className="tip" style={{ left: Math.min(Math.max(hover.x, 100), W - 100), top: hover.y }}>{tooltip(hover.p, hover.s)}</div>
      )}
    </div>
  )
}

/** series: [{ label, color, values: {axis: 0..10}, dashed, fillOpacity }] */
export function Radar({ axes, series, size = 340, center, spin, labels = AXIS_LABEL, showValues }) {
  const R = size / 2 - 70, cx = size / 2, cy = size / 2
  const ang = i => -Math.PI / 2 + (i / axes.length) * 2 * Math.PI
  const pt = (i, v) => [cx + Math.cos(ang(i)) * R * (v / 10), cy + Math.sin(ang(i)) * R * (v / 10)]
  return (
    <div className="radar-wrap">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ maxWidth: '100%', height: 'auto', overflow: 'visible' }}>
        <g className={spin ? 'radar-spin' : ''} opacity=".9">
          <circle cx={cx} cy={cy} r={R + 10} fill="none" stroke="rgba(217,154,108,.18)" strokeDasharray="2 6" />
        </g>
        {[2, 4, 6, 8, 10].map(l => (
          <polygon key={l} points={axes.map((_, i) => pt(i, l).join(',')).join(' ')} fill="none" stroke="rgba(237,232,226,.1)" />
        ))}
        {axes.map((a, i) => {
          const [x, y] = pt(i, 10)
          const [lx, ly] = pt(i, 11.8)
          const anchor = Math.abs(lx - cx) < 8 ? 'middle' : lx > cx ? 'start' : 'end'
          return (
            <g key={a}>
              <line x1={cx} y1={cy} x2={x} y2={y} stroke="rgba(237,232,226,.12)" />
              <text className="radar-label" x={lx} y={ly + 4} textAnchor={anchor}>{labels[a] || a}</text>
            </g>
          )
        })}
        <g className="radar-grow" key={series.map(s => JSON.stringify(s.values)).join('|')}>
          {series.map((s, si) => {
            if (!s.values) return null
            const pts = axes.map((a, i) => pt(i, Math.max(0, Math.min(10, s.values[a] ?? 0))))
            const poly = pts.map(p => p.join(',')).join(' ')
            return (
              <g key={si}>
                <polygon className="radar-poly" points={poly} fill={s.color} fillOpacity={s.fillOpacity ?? (s.dashed ? 0 : 0.14)}
                  stroke={s.color} strokeWidth={s.dashed ? 1.6 : 2.2} strokeDasharray={s.dashed ? '5 4' : undefined} strokeLinejoin="round" />
                {!s.dashed && pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r="3.4" fill={s.color} stroke="#12100E" strokeWidth="1.5" />)}
                {showValues && si === series.length - 1 && pts.map((p, i) => (
                  <text key={`t${i}`} x={p[0]} y={p[1] - 8} textAnchor="middle" fontSize="10" fill="#EDE8E2" fontFamily="var(--mono)">{(s.values[axes[i]] ?? 0).toFixed(1)}</text>
                ))}
              </g>
            )
          })}
        </g>
        {center && (
          <g>
            <circle cx={cx} cy={cy} r="30" fill="#12100E" stroke="rgba(217,154,108,.45)" />
            <text x={cx} y={cy - 2} textAnchor="middle" fontSize="10" fontWeight="700" letterSpacing=".14em" fill="#D99A6C">{center[0]}</text>
            <text x={cx} y={cy + 11} textAnchor="middle" fontSize="9" letterSpacing=".1em" fill="#A9C4D1">{center[1]}</text>
          </g>
        )}
      </svg>
    </div>
  )
}

/** Persona drift: intended -> observed per axis, as connected points on a 0-10 scale. */
export function DriftPlot({ intended, observed, axes, height }) {
  const [ref, w] = useWidth(240)
  const rowH = 34, padL = 128, padR = 54
  const H = height || axes.length * rowH + 30
  const X = v => padL + (Math.max(0, Math.min(10, v)) / 10) * (w - padL - padR)
  return (
    <div className="chart" ref={ref}>
      {w > 0 && (
        <svg width={w} height={H}>
          {[0, 2.5, 5, 7.5, 10].map(v => (
            <g key={v} className="axis">
              <line className="gridline" x1={X(v)} x2={X(v)} y1={8} y2={H - 22} />
              <text x={X(v)} y={H - 6} textAnchor="middle">{v}</text>
            </g>
          ))}
          {axes.map((a, i) => {
            const y = 20 + i * rowH
            const iv = intended?.[a], ov = observed?.[a]
            const d = ov !== undefined && iv !== undefined ? ov - iv : null
            const col = d === null ? '#7A7168' : Math.abs(d) < 1 ? '#5FB37A' : Math.abs(d) < 2.5 ? '#D9A441' : '#D96459'
            return (
              <g key={a} style={{ animation: `rise .5s ${i * 70}ms both` }}>
                <text x={0} y={y + 4} fontSize="12" fill="#A8A097">{AXIS_LABEL[a] || a}</text>
                <line x1={X(0)} x2={X(10)} y1={y} y2={y} stroke="rgba(237,232,226,.08)" strokeWidth="6" strokeLinecap="round" />
                {iv !== undefined && ov !== undefined && (
                  <line x1={X(iv)} x2={X(ov)} y1={y} y2={y} stroke={col} strokeWidth="2.5" strokeLinecap="round" opacity=".85" />
                )}
                {iv !== undefined && <circle cx={X(iv)} cy={y} r="6" fill="#12100E" stroke="#D99A6C" strokeWidth="2" />}
                {ov !== undefined && <circle cx={X(ov)} cy={y} r="5" fill={col} />}
                {d !== null && <text x={w - 2} y={y + 4} textAnchor="end" fontSize="11.5" fill={col} fontFamily="var(--mono)">{d >= 0 ? '+' : ''}{d.toFixed(1)}</text>}
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}

/** Animated 0-10 axis bars with an optional intended marker. */
export function TraitBars({ rows }) {
  return (
    <div>
      {rows.map((r, i) => {
        const col = r.color || fidelityColor(r.value * 10)
        return (
          <div className="tbar" key={r.key} style={{ '--i': i }}>
            <span className="muted">{r.label}</span>
            <div className="track">
              <div className="fill" style={{ width: `${Math.max(0, Math.min(10, r.value)) * 10}%`, background: col }} />
              {r.marker !== undefined && <span className="marker" style={{ left: `calc(${r.marker * 10}% - 1px)` }} title="intended" />}
            </div>
            <span className="mono small" style={{ textAlign: 'right', color: col }}>{r.value.toFixed(1)}</span>
          </div>
        )
      })}
    </div>
  )
}

export function Spark({ values, color = '#D99A6C', width = 120, height = 34 }) {
  if (!values || values.length < 2) return null
  const lo = Math.min(...values), hi = Math.max(...values)
  const pts = values.map((v, i) => [2 + (i / (values.length - 1)) * (width - 4), 2 + (1 - (v - lo) / (hi - lo || 1)) * (height - 4)])
  return (
    <svg width={width} height={height}><path d={smoothPath(pts)} fill="none" stroke={color} strokeWidth="1.8" /></svg>
  )
}
