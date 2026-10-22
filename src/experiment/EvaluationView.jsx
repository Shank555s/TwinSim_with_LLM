// Fidelity result: overall score, per-axis strengths ("why?"), radar, failure modes.
import { Radar } from '../charts/charts.jsx'
import { FailureTags } from '../components/ui.jsx'
import { AXES, FAILURE_LABELS, fmt } from '../constants.js'

const SHORT = { openness: 'O', conscientiousness: 'C', extraversion: 'E', agreeableness: 'A', neuroticism: 'N', political: 'Politics', style: 'Style' }

function strength(v) {
  if (v >= 8) return { w: 'strong', c: 'var(--good)' }
  if (v >= 6) return { w: 'moderate', c: 'var(--warn)' }
  return { w: 'weak', c: 'var(--bad)' }
}

export default function EvaluationView({ evaluation, compact }) {
  if (!evaluation) return null
  const e = evaluation
  const traitAxes = AXES.filter((a) => a.k !== 'style').map((a) => ({ ...a, short: SHORT[a.k] }))
  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="row" style={{ alignItems: 'baseline', gap: 16 }}>
        <div>
          <div style={{ fontFamily: 'var(--serif)', fontSize: compact ? 34 : 44, lineHeight: 1 }}>{fmt(e.overall)}</div>
          <div className="faint">overall fidelity / 100</div>
        </div>
        <div className="spacer" />
        <FailureTags codes={e.failure_codes} />
      </div>
      {!e.valid && <p className="sub">The response was unreadable or looping, so there is nothing to measure. That is scored 0.</p>}
      {e.valid !== false && (
        <div className={compact ? 'stack' : 'grid g2'} style={{ alignItems: 'start' }}>
          <div className="stack" style={{ gap: 6 }}>
            <h3>Why this score</h3>
            {AXES.map((a) => {
              const v = e.axes[a.k]
              const s = strength(v)
              const isTrait = a.k !== 'style'
              return (
                <div key={a.k} style={{ display: 'grid', gridTemplateColumns: '150px 70px 1fr', gap: 8, fontSize: 13, alignItems: 'baseline' }}>
                  <span className="sub">{a.l}</span>
                  <span style={{ color: s.c }}>{s.w} {fmt(v)}</span>
                  <span className="faint">
                    {isTrait ? <>intended <span style={{ color: 'var(--persona)' }}>{fmt(e.intended[a.k])}</span>, expressed <span style={{ color: 'var(--twin)' }}>{fmt(e.observed[a.k])}</span></> : 'surface style vs the communication-style description'}
                  </span>
                </div>
              )
            })}
            {e.failures?.length > 0 && (
              <div className="stack" style={{ gap: 4, marginTop: 8 }}>
                {e.failures.map((f, i) => (
                  <p key={i} className="sub"><span style={{ color: 'var(--bad)' }}>{FAILURE_LABELS[f.code] || f.code}:</span> {f.detail}</p>
                ))}
              </div>
            )}
          </div>
          {!compact && (
            <Radar axes={traitAxes} a={traitAxes.map((a) => e.intended[a.k])} b={traitAxes.map((a) => e.observed[a.k])}
              aName="Persona (intended)" bName="Twin (expressed)" />
          )}
        </div>
      )}
      {!compact && (
        <details>
          <summary>Evaluator evidence</summary>
          <ul className="sub" style={{ margin: '8px 0 0', paddingLeft: 18 }}>
            {e.explanation.map((x, i) => <li key={i}>{x}</li>)}
          </ul>
          <p className="faint" style={{ marginTop: 6 }}>{e.evaluator} · {e.evaluator_kind}</p>
        </details>
      )}
    </div>
  )
}
