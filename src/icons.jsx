// One consistent icon set: 24px grid, 1.7 stroke, round caps. Hand-drawn so no icon dependency is needed.
const P = {
  dashboard: <><rect x="3" y="3" width="7.5" height="9" rx="2" /><rect x="13.5" y="3" width="7.5" height="5.5" rx="2" /><rect x="13.5" y="11.5" width="7.5" height="9.5" rx="2" /><rect x="3" y="15" width="7.5" height="6" rx="2" /></>,
  twin: <><circle cx="9" cy="8" r="3.4" /><circle cx="15.5" cy="8" r="3.4" opacity=".55" /><path d="M3.5 20c.6-3.4 2.8-5.4 5.5-5.4s4.9 2 5.5 5.4" /><path d="M14.6 14.7c.3-.1.6-.1.9-.1 2.7 0 4.9 2 5.5 5.4" opacity=".55" /></>,
  flask: <><path d="M9.5 3h5" /><path d="M10.5 3v5.6L5 18.2A2 2 0 0 0 6.8 21h10.4a2 2 0 0 0 1.8-2.8L13.5 8.6V3" /><path d="M7.5 14.5h9" /></>,
  grid: <><path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4z" /><path d="M14 17h6M17 14v6" /></>,
  brain: <><path d="M9 4.5a3 3 0 0 0-3 3 3 3 0 0 0-2 5.2A3 3 0 0 0 6.5 18 3 3 0 0 0 12 19.5V5.6A2.6 2.6 0 0 0 9 4.5Z" /><path d="M15 4.5a3 3 0 0 1 3 3 3 3 0 0 1 2 5.2 3 3 0 0 1-2.5 5.3A3 3 0 0 1 12 19.5" /><path d="M8.5 9.5c1 .2 1.8 1 2 2M15.5 9.5c-1 .2-1.8 1-2 2M8 14.5h1.5M14.5 14.5H16" /></>,
  dna: <><path d="M7 3c0 4.5 10 4.5 10 9s-10 4.5-10 9" /><path d="M17 3c0 4.5-10 4.5-10 9s10 4.5 10 9" /><path d="M8.6 6.5h6.8M8.6 17.5h6.8" /></>,
  alert: <><path d="M12 3.5 2.8 19a1.3 1.3 0 0 0 1.1 2h16.2a1.3 1.3 0 0 0 1.1-2L12 3.5Z" /><path d="M12 10v4.5M12 17.6v.1" /></>,
  layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 13 9 5 9-5" /><path d="m3 17.5 9 4.5 9-4.5" opacity=".55" /></>,
  history: <><path d="M3.5 12a8.5 8.5 0 1 0 2.5-6" /><path d="M3.5 4v4.5H8" /><path d="M12 8v4.3l3 1.8" /></>,
  report: <><path d="M6 3h8l5 5v11.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 5 19.5v-15A1.5 1.5 0 0 1 6 3Z" /><path d="M14 3v5h5" /><path d="M8.5 17v-3M12 17v-5.5M15.5 17v-2" /></>,
  bell: <><path d="M6 9.5a6 6 0 0 1 12 0c0 6 2.5 7.5 2.5 7.5h-17S6 15.5 6 9.5Z" /><path d="M10 20.5a2.2 2.2 0 0 0 4 0" /></>,
  settings: <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-1-1.5 1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.6 1.6 0 0 0 1.5-1 1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1Z" /></>,
  chevronL: <path d="m14.5 6-6 6 6 6" />,
  chevronR: <path d="m9.5 6 6 6-6 6" />,
  chevronD: <path d="m6 9.5 6 6 6-6" />,
  play: <path d="M7 4.8v14.4a1 1 0 0 0 1.5.9l11.3-7.2a1 1 0 0 0 0-1.7L8.5 3.9A1 1 0 0 0 7 4.8Z" />,
  stop: <rect x="6" y="6" width="12" height="12" rx="2" />,
  sparkle: <><path d="M12 3.5 13.8 9a2 2 0 0 0 1.3 1.3l5.4 1.7-5.4 1.8a2 2 0 0 0-1.3 1.3L12 20.5l-1.8-5.4a2 2 0 0 0-1.3-1.3L3.5 12l5.4-1.7A2 2 0 0 0 10.2 9L12 3.5Z" /></>,
  bolt: <path d="M13 2.5 4.5 13.5H12l-1 8 8.5-11H12l1-8Z" />,
  check: <path d="m5 12.5 4.5 4.5L19 7.5" />,
  checkCircle: <><circle cx="12" cy="12" r="9" /><path d="m8 12.3 2.8 2.8L16.2 9.6" /></>,
  x: <path d="M6 6l12 12M18 6 6 18" />,
  xCircle: <><circle cx="12" cy="12" r="9" /><path d="m9 9 6 6M15 9l-6 6" /></>,
  warn: <><circle cx="12" cy="12" r="9" /><path d="M12 7.5v5.5M12 16.4v.1" /></>,
  info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v5.5M12 7.6v.1" /></>,
  copy: <><rect x="8.5" y="8.5" width="12" height="12" rx="2.2" /><path d="M15.5 8.5V5.7a2.2 2.2 0 0 0-2.2-2.2H5.7a2.2 2.2 0 0 0-2.2 2.2v7.6a2.2 2.2 0 0 0 2.2 2.2h2.8" /></>,
  replay: <><path d="M4 12a8 8 0 1 0 2.4-5.7" /><path d="M4 4v4.5h4.5" /><path d="m10.5 9.3 4.2 2.7-4.2 2.7Z" /></>,
  gauge: <><path d="M4.2 17.5a9 9 0 1 1 15.6 0" /><path d="m12 13.5 4-5" /><circle cx="12" cy="14" r="1.6" /></>,
  compare: <><path d="M8 4v16M16 4v16" /><path d="m4.5 8 3.5-4 3.5 4M12.5 16l3.5 4 3.5-4" /></>,
  save: <><path d="M5 3.5h11l3.5 3.5v12A1.5 1.5 0 0 1 18 20.5H6A1.5 1.5 0 0 1 4.5 19V5A1.5 1.5 0 0 1 5 3.5Z" /><path d="M8 3.5v5h7v-5M8 20.5v-6h8v6" /></>,
  download: <><path d="M12 3.5v12" /><path d="m7 10.5 5 5 5-5" /><path d="M4 20.5h16" /></>,
  cpu: <><rect x="6" y="6" width="12" height="12" rx="2" /><rect x="9.5" y="9.5" width="5" height="5" rx="1" /><path d="M9 2.5V6M15 2.5V6M9 18v3.5M15 18v3.5M2.5 9H6M2.5 15H6M18 9h3.5M18 15h3.5" /></>,
  database: <><ellipse cx="12" cy="5.5" rx="7.5" ry="2.8" /><path d="M4.5 5.5v13c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 7.5-2.8v-13" /><path d="M4.5 12c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 7.5-2.8" /></>,
  target: <><circle cx="12" cy="12" r="8.5" /><circle cx="12" cy="12" r="4.8" /><circle cx="12" cy="12" r="1.2" /></>,
  activity: <path d="M3 12h4l2.5-6.5 5 13 2.5-6.5h4" />,
  trend: <><path d="m3 17 6-6 4 4 8-8" /><path d="M15 7h6v6" /></>,
  trendDown: <><path d="m3 7 6 6 4-4 8 8" /><path d="M15 17h6v-6" /></>,
  shield: <><path d="M12 3 4.5 6v5.5c0 4.6 3.2 8.4 7.5 9.5 4.3-1.1 7.5-4.9 7.5-9.5V6L12 3Z" /><path d="m9 12 2.2 2.2L15.5 10" /></>,
  user: <><circle cx="12" cy="8" r="4" /><path d="M4.5 21c.7-4 3.7-6.5 7.5-6.5s6.8 2.5 7.5 6.5" /></>,
  coins: <><ellipse cx="9" cy="7" rx="6" ry="3" /><path d="M3 7v5c0 1.7 2.7 3 6 3s6-1.3 6-3V7" /><path d="M9 15v2c0 1.7 2.7 3 6 3s6-1.3 6-3v-5c0-1.6-2.4-2.9-5.6-3" /></>,
  landmark: <><path d="M3 21h18M4 10h16M12 3l8.5 5h-17L12 3Z" /><path d="M6 10v8M10 10v8M14 10v8M18 10v8" /></>,
  heart: <path d="M12 20.5s-8.5-4.9-8.5-11.2A4.8 4.8 0 0 1 12 6.4a4.8 4.8 0 0 1 8.5 2.9c0 6.3-8.5 11.2-8.5 11.2Z" />,
  chat: <><path d="M20.5 12a8 8 0 0 1-11.8 7L3.5 20.5 5 15.6A8 8 0 1 1 20.5 12Z" /><path d="M8.5 12h.1M12 12h.1M15.5 12h.1" /></>,
  plus: <path d="M12 5v14M5 12h14" />,
  trash: <><path d="M4 6.5h16M9.5 6.5V4.5h5v2M6.5 6.5l1 13a1.5 1.5 0 0 0 1.5 1.5h6a1.5 1.5 0 0 0 1.5-1.5l1-13" /></>,
  search: <><circle cx="11" cy="11" r="6.5" /><path d="m20 20-4.4-4.4" /></>,
  filter: <path d="M4 5h16l-6.2 7.5V19l-3.6 1.5v-8L4 5Z" />,
  rollback: <><path d="M9 14 4 9l5-5" /><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11" /></>,
  star: <path d="m12 3.5 2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9L12 3.5Z" />,
  eye: <><path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z" /><circle cx="12" cy="12" r="3" /></>,
  eyeOff: <><path d="M10.6 5.6A9.6 9.6 0 0 1 12 5.5c6 0 9.5 6.5 9.5 6.5a17 17 0 0 1-2.6 3.4M6.4 6.5C3.9 8.2 2.5 12 2.5 12s3.5 6.5 9.5 6.5c1.7 0 3.2-.5 4.5-1.2" /><path d="m3 3 18 18M9.9 9.9a3 3 0 0 0 4.2 4.2" /></>,
  clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3.2 2" /></>,
  route: <><circle cx="6" cy="18" r="2.5" /><circle cx="18" cy="6" r="2.5" /><path d="M8.5 18H16a3 3 0 0 0 0-6H8a3 3 0 0 1 0-6h7.5" /></>,
  wand: <><path d="m4 20 11-11M14 5l1-2 1 2 2 1-2 1-1 2-1-2-2-1 2-1ZM19 12l.6-1.4L21 10l-1.4-.6L19 8l-.6 1.4L17 10l1.4.6L19 12Z" /></>,
  scale: <><path d="M12 4v16M7 20h10M5 7h14" /><path d="M5 7 2.5 13a3 3 0 0 0 5 0L5 7ZM19 7l-2.5 6a3 3 0 0 0 5 0L19 7Z" /></>,
  network: <><circle cx="12" cy="5" r="2.2" /><circle cx="5" cy="18" r="2.2" /><circle cx="19" cy="18" r="2.2" /><circle cx="12" cy="12.5" r="2.2" /><path d="M12 7.2v3.1M10.3 13.9l-3.7 2.6M13.7 13.9l3.7 2.6" /></>,
  menu: <path d="M4 7h16M4 12h16M4 17h16" />,
  external: <><path d="M14 4h6v6M20 4l-9 9" /><path d="M18 14v4.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 4 18.5v-11A1.5 1.5 0 0 1 5.5 6H10" /></>,
  motion: <><path d="M4 12h3M17 12h3M12 4v3M12 17v3" /><circle cx="12" cy="12" r="3.2" /></>,
}

export default function Icon({ name, size = 18, stroke = 1.7, className = '', style }) {
  const p = P[name]
  if (!p) return null
  const filled = name === 'play' || name === 'stop'
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={filled ? 'currentColor' : 'none'} stroke="currentColor"
      strokeWidth={filled ? 0 : stroke} strokeLinecap="round" strokeLinejoin="round" className={className} style={style}
      aria-hidden="true">{p}</svg>
  )
}

export function Logo({ size = 22 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="12.5" cy="16" r="7.5" fill="none" stroke="#E8772E" strokeWidth="2.6" />
      <circle cx="19.5" cy="16" r="7.5" fill="none" stroke="#EDE8E2" strokeWidth="2.6" />
    </svg>
  )
}
