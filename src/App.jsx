import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import Icon, { Logo } from './icons.jsx'
import { dateTime, useApi } from './api.js'
import { AppCtx, Dot, Drawer, Toggle } from './ui.jsx'
import Dashboard from './pages/Dashboard.jsx'
import TwinBuilder from './pages/TwinBuilder.jsx'
import ScenarioLab from './pages/ScenarioLab.jsx'
import ExperimentLab from './pages/ExperimentLab.jsx'
import TwinDNA from './pages/TwinDNA.jsx'
import Failures from './pages/Failures.jsx'
import Registry from './pages/Registry.jsx'
import History from './pages/History.jsx'
import Report from './pages/Report.jsx'
import { LearningPill } from './learning.jsx'
import Privacy from './pages/Privacy.jsx'
import Terms from './pages/Terms.jsx'

const NAV = [
  { group: null, items: [{ id: 'dashboard', label: 'Dashboard', icon: 'dashboard', page: Dashboard }] },
  { group: 'Persona', items: [{ id: 'builder', label: 'Twin Builder', icon: 'twin', page: TwinBuilder }] },
  { group: 'Experiment', items: [
    { id: 'scenario', label: 'Scenario Lab', icon: 'flask', page: ScenarioLab },
    { id: 'experiments', label: 'Experiment Lab', icon: 'grid', page: ExperimentLab },
  ] },
  { group: 'Analysis', items: [
    { id: 'dna', label: 'Twin DNA', icon: 'dna', page: TwinDNA },
    { id: 'failures', label: 'Failure Analysis', icon: 'alert', page: Failures },
  ] },
  { group: 'Models', items: [{ id: 'registry', label: 'Model Registry', icon: 'layers', page: Registry }] },
  { group: 'Research', items: [
    { id: 'history', label: 'History', icon: 'history', page: History },
    { id: 'report', label: 'Research Report', icon: 'report', page: Report },
  ] },
]
const PAGES = Object.fromEntries([...NAV.flatMap(g => g.items),
  { id: 'privacy', label: 'Privacy policy', page: Privacy }, { id: 'terms', label: 'Terms of use', page: Terms }].map(i => [i.id, i]))

function readHash() {
  const h = window.location.hash.replace(/^#\/?/, '')
  const [id, q] = h.split('?')
  const params = Object.fromEntries(new URLSearchParams(q || ''))
  return { id: PAGES[id] ? id : 'dashboard', params }
}

const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v) } catch { return d } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)) } catch { /* storage unavailable */ } },
}

// ------------------------------------------------------------------ sidebar
function Sidebar({ page, nav, collapsed, setCollapsed }) {
  const listRef = useRef(null)
  const [ind, setInd] = useState(null)
  useLayoutEffect(() => {
    const nav = listRef.current
    const el = nav?.querySelector(`[data-id="${page}"]`)
    setInd(el ? { y: el.getBoundingClientRect().top - nav.getBoundingClientRect().top + nav.scrollTop, h: el.offsetHeight } : null)
  }, [page, collapsed])
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark"><Logo size={22} /></div>
        <div className="brand-text">
          <div className="brand-name">TwinSim</div>
          <div className="brand-sub">Research Lab</div>
        </div>
      </div>
      <nav className="nav" ref={listRef} style={{ position: 'relative' }}>
        {ind && <div className="nav-active-bg" style={{ transform: `translateY(${ind.y}px)`, top: 0, height: ind.h }} />}
        {NAV.map((g, gi) => (
          <div key={gi} className="nav-group" style={gi === 0 ? { marginTop: 0 } : undefined}>
            {g.group && <div className="nav-label">{g.group}</div>}
            <div className="nav-items">
              {g.items.map(it => (
                <button key={it.id} data-id={it.id} className={`nav-item ${page === it.id ? 'active' : ''}`}
                  onClick={() => nav(it.id)} title={collapsed ? it.label : undefined} aria-current={page === it.id ? 'page' : undefined}>
                  <Icon name={it.icon} size={18} />
                  <span className="nav-text">{it.label}</span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </nav>
      <div className="side-foot">
        <div className="side-links"><a href="#/privacy">Privacy</a><a href="#/terms">Terms</a></div>
        <button className="collapse-btn" onClick={() => setCollapsed(!collapsed)} aria-label="Collapse sidebar">
          <Icon name="chevronL" size={16} /><span className="side-foot-text small">Collapse</span>
        </button>
      </div>
    </aside>
  )
}

// ------------------------------------------------------------------ command center (real /api/status)
const CMD = [
  { key: 'llm', label: 'LLM', map: c => c.state === 'ready' ? ['ONLINE', 'green', true] : c.state === 'untrained' ? ['UNTRAINED', 'amber'] : ['ERROR', 'red'] },
  { key: 'evaluator', label: 'EVALUATOR', map: c => c.state === 'ready' ? ['READY', 'cyan', true] : ['ERROR', 'red'] },
  { key: 'database', label: 'DATABASE', map: c => c.state === 'connected' ? ['CONNECTED', 'green', true] : ['ERROR', 'red'] },
  { key: 'compute', label: 'COMPUTE', map: c => c.state === 'cuda' ? ['CUDA READY', 'green', true] : ['CPU MODE', 'cyan'] },
  { key: 'training', label: 'LEARNING', map: c => c.state === 'busy' ? ['UPDATING', 'violet', true] : ['IDLE', ''] },
]

function CommandCenter({ status, error }) {
  return (
    <div className="command" role="status" aria-label="System status">
      {CMD.map(c => {
        const comp = status?.components?.[c.key]
        const [v, tone, pulse] = error ? ['OFFLINE', 'red'] : comp ? c.map(comp) : ['…', '']
        return (
          <div className="cmd" key={c.key} title={comp?.detail}>
            <Dot tone={tone} pulse={pulse} />
            <div className="grow" style={{ minWidth: 0 }}>
              <div className="row between gap-6">
                <span className="cmd-k">{c.label}</span>
                <span className={`cmd-v ${tone}`}>{v}</span>
              </div>
              <div className="cmd-d">{error ? 'backend unreachable' : comp?.detail || ' '}</div>
              {c.key === 'training' && comp?.state === 'busy' && comp.progress !== null && (
                <div className="bar"><i style={{ width: `${Math.round(comp.progress * 100)}%` }} /></div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ------------------------------------------------------------------ notifications / settings
const NOTIFY_TYPES = { model_promoted: ['checkCircle', 'green'], model_rejected: ['xCircle', 'amber'], job_failed: ['alert', 'red'],
  interrupted: ['warn', 'amber'], model_activated: ['layers', 'violet'], model_rollback: ['rollback', 'violet'],
  benchmark_completed: ['gauge', 'cyan'], learning_experiment: ['trend', 'violet'], model_baseline: ['sparkle', 'cyan'],
  learning_cycle: ['brain', 'violet'], checkpoints_pruned: ['layers', ''] }

function Notifications({ open, onClose, events, nav }) {
  return (
    <Drawer open={open} onClose={onClose}>
      <div className="row between"><h3 className="card-title"><Icon name="bell" size={16} />Notifications</h3>
        <button className="icon-btn" onClick={onClose} aria-label="Close"><Icon name="x" size={16} /></button></div>
      <p className="small dim">Model decisions, benchmarks and failed jobs, taken from the research history log.</p>
      <div className="notif-list mt-12">
        {events.length === 0 && <p className="muted small">Nothing yet.</p>}
        {events.map(e => {
          const [ic, tone] = NOTIFY_TYPES[e.type] || ['info', '']
          return (
            <button key={e.id} className="tl-item" style={{ textAlign: 'left', background: 'rgba(34,30,26,.55)', border: '1px solid var(--border)', cursor: 'pointer', color: 'inherit', font: 'inherit' }}
              onClick={() => { onClose(); nav(e.type.startsWith('model') ? 'registry' : 'history') }}>
              <div className="row top">
                <span className={tone} style={{ marginTop: 2 }}><Icon name={ic} size={16} /></span>
                <div className="grow"><div className="small" style={{ lineHeight: 1.45 }}>{e.summary}</div>
                  <div className="tiny dim mt-4 mono">{dateTime(e.ts)}</div></div>
              </div>
            </button>
          )
        })}
      </div>
    </Drawer>
  )
}

function Settings({ open, onClose, reduce, setReduce }) {
  const sys = useApi(open ? '/system' : null)
  const s = sys.data
  return (
    <Drawer open={open} onClose={onClose}>
      <div className="row between"><h3 className="card-title"><Icon name="settings" size={16} />Settings &amp; system</h3>
        <button className="icon-btn" onClick={onClose} aria-label="Close"><Icon name="x" size={16} /></button></div>
      <div className="card mt-16">
        <div className="row between"><div><div className="small" style={{ fontWeight: 600 }}>Reduce motion</div>
          <div className="tiny dim">Turns off animations. The OS-level setting is always respected.</div></div>
          <Toggle on={reduce} onChange={setReduce} /></div>
      </div>
      {s && (
        <div className="col mt-16 small">
          <div className="section-label">Hardware</div>
          <dl className="kv" style={{ gridTemplateColumns: '120px 1fr' }}>
            <dt>Backend</dt><dd className="mono">{s.hardware.backend}{s.hardware.torch_installed ? ' (torch installed)' : ''}</dd>
            <dt>Device</dt><dd className="mono">{s.hardware.cuda_available ? `CUDA · ${s.hardware.gpu} · ${s.hardware.vram_gb} GB` : `CPU · ${s.hardware.cpu_cores} cores`}</dd>
            <dt>RAM</dt><dd className="mono">{s.hardware.ram_gb} GB</dd>
            <dt>Recommended</dt><dd>{s.recommended.profile}: {s.recommended.reason}</dd>
          </dl>
          <div className="section-label mt-12">Evaluator</div>
          <dl className="kv" style={{ gridTemplateColumns: '120px 1fr' }}>
            <dt>Version</dt><dd className="mono">{s.evaluator.version}</dd>
            <dt>Kind</dt><dd>{s.evaluator.kind}</dd>
            <dt>Weights</dt><dd className="mono">{Object.entries(s.evaluator.weights).map(([k, v]) => `${k.slice(0, 4)} ${v}`).join(' · ')}</dd>
            {s.evaluator.calibration?.oracle !== undefined && (<><dt>Calibration</dt>
              <dd className="mono">oracle {s.evaluator.calibration.oracle} · bland {s.evaluator.calibration.bland} · shuffled {s.evaluator.calibration.shuffled}</dd></>)}
          </dl>
          <div className="section-label mt-12">Data ingestion</div>
          <dl className="kv" style={{ gridTemplateColumns: '120px 1fr' }}>
            <dt>Corpus docs</dt><dd className="mono">{s.ingestion.corpus_documents}</dd>
            <dt>Persona rows</dt><dd className="mono">{s.ingestion.persona_examples.ok} ok · {s.ingestion.persona_examples.skipped} skipped</dd>
          </dl>
          <div className="section-label mt-12">External services</div>
          <p className="small muted" style={{ margin: 0 }}>{s.external_llm ? 'External LLM in use' : 'None. Every response comes from the local TwinSim checkpoint; no API keys are used.'}</p>
        </div>
      )}
    </Drawer>
  )
}

// ------------------------------------------------------------------ app
export default function App() {
  const [route, setRoute] = useState(readHash)
  const [collapsed, setCollapsed] = useState(() => store.get('ts.sidebar', false))
  const [reduce, setReduce] = useState(() => store.get('ts.reduceMotion', false))
  const [toasts, setToasts] = useState([])
  const [drawer, setDrawer] = useState(null)
  const [lastSeen, setLastSeen] = useState(() => store.get('ts.notifSeen', 0))
  const [twinDraft, setTwinDraft] = useState(null)
  const [narrow, setNarrow] = useState(() => window.innerWidth < 900)
  useEffect(() => { const f = () => setNarrow(window.innerWidth < 900); window.addEventListener('resize', f); return () => window.removeEventListener('resize', f) }, [])
  const isCollapsed = collapsed || narrow

  useEffect(() => { const f = () => setRoute(readHash()); window.addEventListener('hashchange', f); return () => window.removeEventListener('hashchange', f) }, [])
  useEffect(() => { window.scrollTo(0, 0) }, [route.id])
  useEffect(() => { store.set('ts.sidebar', collapsed) }, [collapsed])
  useEffect(() => { store.set('ts.reduceMotion', reduce); document.documentElement.classList.toggle('reduce-motion', reduce) }, [reduce])

  const status = useApi('/status', { poll: 2500 })
  const notif = useApi('/history?limit=60', { poll: 15000 })
  const events = (notif.data?.events || []).filter(e => NOTIFY_TYPES[e.type]).slice(0, 25)
  const unread = events.filter(e => e.ts > lastSeen).length

  const nav = useCallback((id, params) => {
    const q = params ? `?${new URLSearchParams(params)}` : ''
    window.location.hash = `/${id}${q}`
  }, [])
  const toast = useCallback((msg, tone = '') => {
    const id = Math.random()
    setToasts(t => [...t, { id, msg, tone }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4200)
  }, [])
  const ctx = useMemo(() => ({ nav, toast, status: status.data, statusError: status.error, refreshStatus: status.reload, twinDraft, setTwinDraft, route }),
    [nav, toast, status.data, status.error, status.reload, twinDraft, route])

  const cur = PAGES[route.id]
  const Page = cur.page
  const st = status.data
  return (
    <AppCtx.Provider value={ctx}>
      <div className={`app ${isCollapsed ? 'collapsed' : ''}`}>
        <Sidebar page={route.id} nav={nav} collapsed={isCollapsed} setCollapsed={v => !narrow && setCollapsed(v)} />
        <div className="main">
          <header className="topbar">
            <div className="crumbs grow"><span>TwinSim</span><Icon name="chevronR" size={13} /><b>{cur.label}</b></div>
            <span className="top-pill hide-sm" title="Active model (from the registry)">
              <Dot tone={st?.active_model ? 'violet' : 'amber'} pulse={!!st?.active_model} />
              {st?.active_model ? <><b className="mono">{st.active_model}</b><span className="dim mono">{st.parameters_h} params</span></> : <span>no model trained</span>}
            </span>
            <LearningPill />
            <span className="top-pill hide-sm" title={st?.components?.compute?.detail}>
              <Icon name="cpu" size={14} />
              <b className="mono">{st?.components?.compute?.state === 'cuda' ? 'CUDA READY' : st ? 'CPU' : '…'}</b>
            </span>
            <button className="icon-btn" aria-label="Notifications" onClick={() => { setDrawer('notif'); const now = Date.now() / 1000; setLastSeen(now); store.set('ts.notifSeen', now) }}>
              <Icon name="bell" size={16} />{unread > 0 && <span className="count">{unread > 9 ? '9+' : unread}</span>}
            </button>
            <button className="icon-btn" aria-label="Settings" onClick={() => setDrawer('settings')}><Icon name="settings" size={16} /></button>
          </header>
          <CommandCenter status={st} error={status.error && !st ? status.error : status.error?.status === 0 ? status.error : null} />
          <main className="page-wrap">
            <div className="page" key={route.id}><Page params={route.params} /></div>
          </main>
        </div>
      </div>
      <Notifications open={drawer === 'notif'} onClose={() => setDrawer(null)} events={events} nav={nav} />
      <Settings open={drawer === 'settings'} onClose={() => setDrawer(null)} reduce={reduce} setReduce={setReduce} />
      <div className="toasts" aria-live="polite">
        {toasts.map(t => (
          <div key={t.id} className={`toast ${t.tone}`}>
            <span className={t.tone === 'ok' ? 'green' : t.tone === 'err' ? 'red' : 'violet'}><Icon name={t.tone === 'ok' ? 'checkCircle' : t.tone === 'err' ? 'xCircle' : 'info'} size={16} /></span>
            <span>{t.msg}</span>
          </div>
        ))}
      </div>
    </AppCtx.Provider>
  )
}

