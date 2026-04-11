/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          0: '#ffffff',
          1: '#f8f9fa',
          2: '#f0f1f3',
          3: '#e1e4e8',
        },
        accent: {
          DEFAULT: '#2563eb',
          dim: '#1d4ed8',
        },
        text: {
          primary: '#1f2937',
          secondary: '#6b7280',
          muted: '#9ca3af',
        },
      },
    },
  },
  plugins: [],
}
