// Hash navigation helper: go('scenario', { persona: 'p1' }) -> #/scenario?persona=p1
export function go(id, params) {
  const q = params ? '?' + new URLSearchParams(params).toString() : ''
  window.location.hash = `#/${id}${q}`
}
