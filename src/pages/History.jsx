import { useMemo, useState } from 'react'
import Icon from '../icons.jsx'
import { clock, useApi } from '../api.js'
import { Badge, Card, Empty, ErrorBox, Loading, PageHead } from '../ui.jsx'

const TYPE_META = {
  experiment_run: ['flask', 'cyan'], experiment_started: ['grid', 'cyan'], experiment_completed: ['grid', 'cyan'], experiment_replayed: ['replay', 'cyan'],
  model_promoted: ['checkCircle', 'green'], model_rejected: ['xCircle', 'amber'], model_baseline: ['sparkle', 'violet'], model_activated: ['layers', 'violet'], model_rollback: ['rollback', 'violet'],
  training_started: ['brain', 'violet'], checkpoint_saved: ['save', 'violet'], dataset_created: ['database', ''], feedback_harvested: ['sparkle', 'violet'],
  examples_retired: ['trash', ''], benchmark_completed: ['gauge', 'cyan'], learning_experiment: ['trend', 'green'], human_evaluation: ['user', 'cyan'],
  persona_created: ['twin', ''], persona_edited: ['twin', ''], persona_deleted: ['twin', ''], scenario_created: ['chat', ''], scenario_deleted: ['chat', ''],
  learning_cycle: ['brain', 'violet'], checkpoints_pruned: ['layers', ''], settings_changed: ['settings', ''], job_failed: ['alert', 'red'], interrupted: ['warn', 'amber'],
}

function dayLabel(ts) {
  const d = new Date(ts * 1000), now = new Date()
  const y = new Date(now); y.setDate(now.getDate() - 1)
  if (d.toDateString() === now.toDateString()) return 'Today'
  if (d.toDateString() === y.toDateString()) return 'Yesterday'
  return d.toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })
}

export default function History() {
  const [type, setType] = useState('')
  const [q, setQ] = useState('')
  const [open, setOpen] = useState(null)
  const h = useApi(`/history?limit=500${type ? `&type=${type}` : ''}`, { poll: 15000 })
  const events = useMemo(() => (h.data?.events || []).filter(e => !q || e.summary.toLowerCase().includes(q.toLowerCase())), [h.data, q])
  const groups = useMemo(() => {
    const g = []
    events.forEach(e => {
      const l = dayLabel(e.ts)
      if (!g.length || g[g.length - 1].label !== l) g.push({ label: l, items: [] })
      g[g.length - 1].items.push(e)
    })
    return g
  }, [events])

  return (<>
    <PageHead eyebrow={<><Icon name="history" size={13} />Research</>} title="History"
      sub="An append-only log of everything that happened: trials, datasets, training runs, checkpoints, benchmark results and model decisions."
      right={<>
        <div className="row" style={{ position: 'relative' }}><Icon name="search" size={15} className="dim" style={{ position: 'absolute', left: 11 }} />
          <input className="input" style={{ paddingLeft: 32, width: 230 }} placeholder="Search events" value={q} onChange={e => setQ(e.target.value)} /></div>
        <select className="select" style={{ width: 210 }} value={type} onChange={e => setType(e.target.value)}>
          <option value="">All event types</option>
          {(h.data?.types || []).map(t => <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>)}
        </select>
      </>} />
    <ErrorBox error={h.error} />
    {h.loading && !h.data && <Card><Loading rows={6} /></Card>}
    {h.data && !events.length && <Card><Empty icon="history" title="No events yet">Run an experiment and it will be logged here.</Empty></Card>}
    {groups.map(g => (
      <section key={g.label} className="day">
        <div className="day-label">{g.label} · {g.items.length}</div>
        <div className="tl">
          {g.items.map((e, i) => {
            const [ic, tone] = TYPE_META[e.type] || ['info', '']
            const isOpen = open === e.id
            return (
              <div key={e.id} className="tl-item fade-in" style={{ animationDelay: `${Math.min(i, 12) * 25}ms`, cursor: e.data ? 'pointer' : 'default' }} onClick={() => e.data && setOpen(isOpen ? null : e.id)}>
                <span className={`tl-dot ${tone}`} />
                <div style={{ minWidth: 0 }}>
                  <div className="row gap-6"><span className={tone || 'dim'} style={{ display: 'inline-flex' }}><Icon name={ic} size={14} /></span>
                    <span className="small" style={{ lineHeight: 1.5 }}>{e.summary}</span></div>
                  <div className="row gap-6 mt-4"><Badge>{e.type.replace(/_/g, ' ')}</Badge>{e.entity_id && <span className="tiny dim mono">{e.entity_type} · {e.entity_id}</span>}</div>
                  {isOpen && <pre className="mono tiny fade-in" style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: 'var(--text-2)', background: 'rgba(18,16,14,.6)', border: '1px solid var(--border)', borderRadius: 10, padding: 10, marginTop: 8, maxHeight: 280, overflow: 'auto' }}>{JSON.stringify(e.data, null, 2)}</pre>}
                </div>
                <span className="tiny dim mono nowrap">{clock(e.ts)}</span>
              </div>
            )
          })}
        </div>
      </section>
    ))}
    {h.data && h.data.total > (h.data.events || []).length && <p className="tiny dim mt-16">Showing the latest {(h.data.events || []).length} of {h.data.total} events.</p>}
  </>)
}
