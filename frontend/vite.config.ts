import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/papers': 'http://localhost:8788',
      '/chat': 'http://localhost:8788',
      '/graph': 'http://localhost:8788',
      '/search': 'http://localhost:8788',
      '/ingest': 'http://localhost:8788',
      '/vault': 'http://localhost:8788',
      '/workspaces': 'http://localhost:8788',
      '/health': 'http://localhost:8788',
    },
  },
})
