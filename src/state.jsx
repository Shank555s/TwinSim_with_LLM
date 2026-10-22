import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { api } from './api/client.js'

const Ctx = createContext(null)

// App-wide data that most pages need: system info, personas, scenarios, model registry.
export function LabProvider({ children }) {
  const [state, set] = useState({ system: null, personas: [], scenarios: [], models: [], active: null, error: null, ready: false })
  const refresh = useCallback(async () => {
    try {
      const [system, p, s, m] = await Promise.all([
        api.get('/api/system'), api.get('/api/personas'), api.get('/api/scenarios'), api.get('/api/models'),
      ])
      set({ system, personas: p.personas, defaultPersona: p.default, scenarios: s.scenarios, models: m.models, active: m.active, error: null, ready: true })
    } catch (error) {
      set((st) => ({ ...st, error, ready: true }))
    }
  }, [])
  useEffect(() => { refresh() }, [refresh])
  return <Ctx.Provider value={{ ...state, refresh }}>{children}</Ctx.Provider>
}

export const useLab = () => useContext(Ctx)
