// Minimal 16px line icons (stroke = currentColor).
const P = {
  dashboard: 'M2 2h5v6H2zM9 2h5v3H9zM9 7h5v7H9zM2 10h5v4H2z',
  twin: 'M5 6a2 2 0 1 0 0-.01M11 6a2 2 0 1 0 0-.01M1.5 14c.5-2.5 2-4 3.5-4s3 1.5 3.5 4M7.5 14c.5-2.5 2-4 3.5-4s3 1.5 3.5 4',
  scenario: 'M2 3h12v8H6l-3 3v-3H2z',
  experiment: 'M6 2h4M7 2v4L3 13a1 1 0 0 0 1 1.5h8A1 1 0 0 0 13 13L9 6V2',
  learning: 'M2 13l4-4 3 2 5-6M10 5h4v4',
  registry: 'M3 3h10v3H3zM3 7.5h10v3H3zM3 12h10v2H3z',
  dna: 'M4 2c0 4 8 4 8 6s-8 2-8 6M12 2c0 4-8 4-8 6s8 2 8 6M5 4h6M5 12h6',
  failure: 'M8 2l6.5 11.5h-13zM8 6.5v3.5M8 11.6v.1',
  history: 'M8 3.5a4.5 4.5 0 1 1-4.5 4.5M3.5 4v4h4M8 5.5V8l2 1.5',
  report: 'M4 2h6l3 3v9H4zM10 2v3h3M6 8h5M6 10.5h5',
  demo: 'M4 3l9 5-9 5z',
}

export default function Icon({ name, size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={P[name]} />
    </svg>
  )
}
