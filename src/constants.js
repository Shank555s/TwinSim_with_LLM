export const TRAITS = [
  { k: 'openness', l: 'Openness', lo: 'conventional', hi: 'exploratory' },
  { k: 'conscientiousness', l: 'Conscientiousness', lo: 'spontaneous', hi: 'organised' },
  { k: 'extraversion', l: 'Extraversion', lo: 'reserved', hi: 'outgoing' },
  { k: 'agreeableness', l: 'Agreeableness', lo: 'sceptical', hi: 'trusting' },
  { k: 'neuroticism', l: 'Neuroticism', lo: 'stable', hi: 'sensitive' },
]

export const AXES = [
  { k: 'openness', l: 'Openness' },
  { k: 'conscientiousness', l: 'Conscientiousness' },
  { k: 'extraversion', l: 'Extraversion' },
  { k: 'agreeableness', l: 'Agreeableness' },
  { k: 'neuroticism', l: 'Neuroticism' },
  { k: 'political', l: 'Political alignment' },
  { k: 'style', l: 'Communication style' },
]

export const FAILURE_LABELS = {
  exaggeration: 'Generative exaggeration',
  blue_shift: 'Blue-shift bias',
  stereotype: 'Stereotype projection',
  inconsistency: 'Temporal inconsistency',
  red_shift: 'Red-shift bias',
  degeneration: 'Degenerate output',
}

export const FAILURE_HELP = {
  exaggeration: 'A trait is expressed far more extremely than the persona has it.',
  blue_shift: 'The answer leans further left than the persona.',
  stereotype: 'The answer follows a demographic archetype instead of this individual.',
  inconsistency: 'The answer contradicts itself (supports and opposes, or mixed trait signals).',
  red_shift: 'The answer leans further right than the persona.',
  degeneration: 'Empty, unreadable or looping text; scored 0 because nothing can be measured.',
}

export const DOMAINS = {
  economic: 'Economic decision',
  political: 'Political opinion',
  health: 'Health choice',
  social: 'Social media reply',
}

export const SOURCE_LABELS = {
  synthetic: 'Synthetic corpus',
  evaluator_feedback: 'Evaluator feedback',
  auto_best_of_n: 'Auto-generated (best of N)',
  human_rating: 'Human rating',
  human_correction: 'Human correction',
}

export const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d))
export const pct = (v, d = 0) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(d)}%`)
export const signed = (v, d = 1) => (v === null || v === undefined ? '—' : `${v > 0 ? '+' : ''}${Number(v).toFixed(d)}`)
export const when = (ts) => (ts ? new Date(ts * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—')
