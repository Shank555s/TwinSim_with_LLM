import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The FastAPI backend (backend/main.py) runs on :8000. No external model APIs are used.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
