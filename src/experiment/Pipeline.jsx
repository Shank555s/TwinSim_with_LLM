// The experiment pipeline, lit up from the real job stage reported by the backend.
import { Progress } from '../components/ui.jsx'

export const FULL_PIPELINE = [
  { k: 'persona', l: 'Persona' },
  { k: 'scenario', l: 'Scenario' },
  { k: 'llm', l: 'TwinSim LLM' },
  { k: 'response', l: 'Response' },
  { k: 'evaluate', l: 'Fidelity evaluation' },
  { k: 'feedback', l: 'Feedback' },
  { k: 'replay', l: 'Replay buffer' },
  { k: 'train', l: 'Training' },
  { k: 'model', l: 'New model' },
]

// Map a backend stage string to the index of the pipeline node it represents.
export function stageToIndex(stage, steps = FULL_PIPELINE) {
  const s = (stage || '').toLowerCase()
  const find = (k) => steps.findIndex((x) => x.k === k)
  if (s === 'done') return steps.length
  if (s.startsWith('generate') || s.startsWith('starting') || s === 'queued') return find('llm')
  if (s.startsWith('evaluate')) return find('evaluate')
  if (s.startsWith('feedback')) return find('feedback')
  if (s.startsWith('data') || s.startsWith('tokenizer') || s.startsWith('model')) return find('replay')
  if (s.startsWith('training') || s.startsWith('stage')) return find('train')
  if (s.startsWith('rerun') && find('rerun') >= 0) return find('rerun')
  if (s.startsWith('benchmark') || s.startsWith('rerun') || s.startsWith('iteration') || s.startsWith('baseline')) return find('model')
  return -1
}

export default function Pipeline({ job, steps = FULL_PIPELINE }) {
  const status = job?.status
  const idx = status === 'completed' ? steps.length : stageToIndex(job?.stage, steps)
  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="pipeline" style={{ '--n': steps.length }}>
        {steps.map((st, i) => {
          let cls = ''
          if (i < idx) cls = 'done'
          else if (i === idx && status === 'running') cls = 'active'
          if (i === idx && status === 'failed') cls = 'failed'
          return (
            <div key={st.k} className={`pstage ${cls}`} aria-current={cls === 'active' ? 'step' : undefined}>
              <div className="node" />
              {st.l}
            </div>
          )
        })}
      </div>
      {job && (
        <div className="stack" style={{ gap: 6 }}>
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <span className="sub">{job.stage}{job.message ? `: ${job.message}` : ''}</span>
            <span className="faint">{job.status}{job.progress != null ? ` · ${Math.round(job.progress * 100)}%` : ''}</span>
          </div>
          <Progress value={job.status === 'completed' ? 1 : job.progress} />
          {job.status === 'failed' && <pre className="code" style={{ color: 'var(--bad)' }}>{job.error}</pre>}
        </div>
      )}
    </div>
  )
}
