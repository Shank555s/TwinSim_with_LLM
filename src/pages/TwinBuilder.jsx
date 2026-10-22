import { Fragment, useEffect, useMemo, useState } from 'react'
import Icon from '../icons.jsx'
import { api, useApi } from '../api.js'
import { Radar } from '../charts.jsx'
import { Badge, Btn, Card, CardHead, ErrorBox, Field, Loading, Note, PageHead, Slider, useApp } from '../ui.jsx'
import { PersonaPicker } from '../components.jsx'

const TRAIT_DEFS = [
  ['openness', 'Openness', 'Conventional', 'Curious'],
  ['conscientiousness', 'Conscientiousness', 'Spontaneous', 'Organised'],
  ['extraversion', 'Extraversion', 'Reserved', 'Outgoing'],
  ['agreeableness', 'Agreeableness', 'Challenging', 'Cooperative'],
  ['neuroticism', 'Neuroticism', 'Calm', 'Anxious'],
]
const VIEW_AXES = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'political', 'environmental']
const VIEW_LABELS = { openness: 'Openness', conscientiousness: 'Conscientiousness', extraversion: 'Extraversion', agreeableness: 'Agreeableness',
  neuroticism: 'Neuroticism', political: 'Politics', environmental: 'Environment' }
const TEXT = [
  ['label', 'Twin name', 'e.g. Rural nurse, 52'], ['age', 'Age', '34'], ['gender', 'Gender', 'female'],
  ['education', 'Education', "bachelor's degree"], ['occupation', 'Occupation', 'high school teacher'],
  ['location', 'Location', 'suburban Ohio, USA'],
]
const level = v => (v < 40 ? 'Low' : v > 60 ? 'High' : 'Moderate')
const lean = v => (v < 20 ? 'Strongly progressive' : v < 40 ? 'Leans progressive' : v <= 60 ? 'Centrist' : v <= 80 ? 'Leans conservative' : 'Strongly conservative')

// Plain-language reading of the sliders (a description of the input, not a measurement).
function signature(p) {
  const out = []
  TRAIT_DEFS.forEach(([k, label, lo, hi]) => {
    const v = p[k]
    if (v > 60) out.push([hi, 'v'])
    else if (v < 40) out.push([lo, 'v'])
    else out.push([`Moderate ${label.toLowerCase()}`, ''])
  })
  out.push([lean(p.political), 'c'])
  out.push([p.environmental > 60 ? 'Environmentally concerned' : p.environmental < 40 ? 'Environmentally sceptical' : 'Environmentally neutral', 'c'])
  return out
}

export default function TwinBuilder({ params }) {
  const { nav, toast, setTwinDraft, status } = useApp()
  const list = useApi('/personas')
  const [form, setForm] = useState(null)
  const [srcId, setSrcId] = useState(params?.persona || null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)

  const personas = list.data?.personas || []
  const source = personas.find(p => p.id === srcId)

  useEffect(() => {
    if (!list.data || form) return
    const start = personas.find(p => p.id === params?.persona)
    setForm(start ? { ...start } : { ...list.data.default })
  }, [list.data]) // eslint-disable-line react-hooks/exhaustive-deps

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))
  const radar = useMemo(() => form && Object.fromEntries(VIEW_AXES.map(a => [a, (form[a] ?? 50) / 10])), [form])
  const sig = form ? signature(form) : []
  const editable = source && !source.is_benchmark && !status?.public_mode

  const load = id => { const p = personas.find(x => x.id === id); setSrcId(id); setForm({ ...p }); setErr(null) }
  const fresh = () => { setSrcId(null); setForm({ ...list.data.default, label: '' }); setErr(null) }

  async function save(asNew) {
    setBusy(true); setErr(null)
    try {
      const body = { ...form }; delete body.id; delete body.is_benchmark; delete body.source
      const saved = editable && !asNew ? await api(`/personas/${srcId}`, { method: 'PUT', body }) : await api('/personas', { method: 'POST', body })
      await list.reload(true)
      setSrcId(saved.id); setForm({ ...saved })
      toast(`Twin "${saved.label}" saved`, 'ok')
      return saved
    } catch (e) { setErr(e); return null } finally { setBusy(false) }
  }
  async function remove() {
    if (!editable) return
    setBusy(true)
    try { await api(`/personas/${srcId}`, { method: 'DELETE' }); await list.reload(true); toast('Twin archived'); fresh() } catch (e) { setErr(e) } finally { setBusy(false) }
  }
  async function run() {
    if (source && JSON.stringify(stripMeta(source)) === JSON.stringify(stripMeta(form))) { nav('scenario', { persona: source.id }); return }
    setTwinDraft({ ...form, id: undefined })
    nav('scenario', { persona: 'draft' })
  }

  if (list.error) return <ErrorBox error={list.error} />
  if (!form) return <Card><Loading rows={6} /></Card>

  return (<>
    <PageHead eyebrow={<><Icon name="twin" size={13} />Persona</>} title="Create a twin"
      sub="Describe a person. TwinSim serialises this profile into the prompt of its own Transformer, and the evaluator later checks whether the answers express these traits."
      right={<>
        <Btn icon="plus" kind="ghost" onClick={fresh}>New</Btn>
        <Btn icon="save" onClick={() => save(false)} loading={busy}>{editable ? 'Save changes' : 'Save twin'}</Btn>
        {source && <Btn icon="copy" kind="ghost" onClick={() => save(true)} disabled={busy}>Save as new</Btn>}
        <Btn kind="primary" icon="flask" onClick={run}>Run in Scenario Lab</Btn>
      </>} />

    <div className="grid g-main">
      <div className="col gap-16">
        <Card>
          <CardHead icon="user" title="Start from a saved twin" sub="Benchmark personas are fixed (they anchor the benchmark); duplicate one to change it." />
          <PersonaPicker personas={personas} value={srcId} onChange={load} />
        </Card>

        <Card className="stagger">
          <CardHead icon="user" title="Identity" sub={source ? <>Editing <b>{source.label}</b>{source.is_benchmark ? ' · read-only benchmark persona, saving creates a copy' : ''}</> : 'A new twin'} />
          <div className="grid g3">
            {TEXT.map(([k, label, ph]) => (
              <Field key={k} label={label}><input className="input" value={form[k] ?? ''} placeholder={ph} onChange={e => set(k, e.target.value)} /></Field>
            ))}
          </div>
          <div className="grid g2 mt-12">
            <Field label="Economic situation"><input className="input" value={form.economic ?? ''} onChange={e => set('economic', e.target.value)} placeholder="worried about rising costs" /></Field>
            <Field label="Communication style" hint="Parsed by the evaluator's style rubric (e.g. formal, warm, blunt, analytical, informal)">
              <input className="input" value={form.commStyle ?? ''} onChange={e => set('commStyle', e.target.value)} placeholder="thoughtful, formal, occasionally humorous" />
            </Field>
          </div>
          <div className="mt-12"><Field label="Notes"><textarea className="textarea" rows={2} value={form.notes ?? ''} onChange={e => set('notes', e.target.value)} placeholder="has two children; volunteers at the library" /></Field></div>
        </Card>

        <Card>
          <CardHead icon="brain" title="Personality" sub="Big Five (0–100), the axes the rubric evaluator measures" />
          {TRAIT_DEFS.map(([k, label, lo, hi]) => (
            <div key={k}>
              <Slider label={label} value={form[k] ?? 50} onChange={v => set(k, v)} />
              <div className="slider-ends"><span>{lo}</span><span>{hi}</span></div>
            </div>
          ))}
          <div className="divider" />
          <Slider label="Politics" value={form.political ?? 50} onChange={v => set('political', v)} color="var(--measure)" />
          <div className="slider-ends"><span>Progressive</span><span>Conservative</span></div>
          <Slider label="Environment" value={form.environmental ?? 50} onChange={v => set('environmental', v)} color="var(--measure)" />
          <div className="slider-ends"><span>Sceptical</span><span>Concerned</span></div>
        </Card>
        <ErrorBox error={err} />
      </div>

      <div>
        <Card glow="v" className="profile-card">
          <div className="row between">
            <span className="upper violet">Twin profile</span>
            {source?.is_benchmark ? <Badge tone="cyan">benchmark</Badge> : source ? <Badge tone="violet">saved</Badge> : <Badge tone="amber">unsaved</Badge>}
          </div>
          <div className="mt-12" style={{ fontSize: 20, fontWeight: 700, letterSpacing: '-.02em' }}>{form.label || 'Unnamed twin'}</div>
          <div className="small muted">{[form.age && `${form.age} y/o`, form.gender, form.occupation].filter(Boolean).join(' · ') || 'Fill in the identity fields'}</div>
          <div className="small dim">{form.location}</div>
          <Radar axes={VIEW_AXES} labels={VIEW_LABELS} size={320} center={['TWIN', 'PROFILE']}
            series={[{ label: 'Intended', color: '#E8772E', values: radar }]} />
          <div className="section-label">Behavioral signature</div>
          <div className="sig">{sig.map(([t, tone], i) => <span key={i} className={`chip ${tone}`}>{t}</span>)}</div>
          <div className="divider" />
          <dl className="kv" style={{ gridTemplateColumns: '110px 1fr' }}>
            {TRAIT_DEFS.map(([k, label]) => <Fragment key={k}><dt>{label}</dt><dd className="mono">{level(form[k])} · {(form[k] / 100).toFixed(2)}</dd></Fragment>)}
            <dt>Politics</dt><dd className="mono">{lean(form.political)} · {form.political}</dd>
            <dt>Style</dt><dd>{form.commStyle || '–'}</dd>
          </dl>
          <div className="mt-12"><Note>This preview restates your inputs. Whether the twin actually behaves this way is measured in the Scenario Lab.</Note></div>
          {editable && <div className="mt-12"><Btn size="sm" kind="danger" icon="trash" onClick={remove} disabled={busy}>Archive twin</Btn></div>}
        </Card>
      </div>
    </div>
  </>)
}

function stripMeta(p) {
  const o = { ...p }
  ;['id', 'is_benchmark', 'source', 'created_at', 'updated_at'].forEach(k => delete o[k])
  return o
}
