import { CSSProperties } from 'react'

interface Option<T extends string> {
  value: T
  label: string
}

interface PillToggleProps<T extends string> {
  options: Option<T>[]
  value: T
  onChange: (v: T) => void
  style?: CSSProperties
}

export function PillToggle<T extends string>({
  options,
  value,
  onChange,
  style,
}: PillToggleProps<T>) {
  return (
    <div className="pill" style={{ width: '100%', ...style }}>
      {options.map(opt => (
        <button
          key={opt.value}
          className="pill-tab"
          data-active={value === opt.value ? 'true' : 'false'}
          onClick={() => onChange(opt.value)}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}
