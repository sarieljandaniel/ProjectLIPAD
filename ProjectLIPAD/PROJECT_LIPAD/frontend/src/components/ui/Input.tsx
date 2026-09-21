import { InputHTMLAttributes } from "react";

type Props = InputHTMLAttributes<HTMLInputElement> & {
  label: string;
};

export function Input({ label, className = "", id, ...rest }: Props) {
  const inputId = id || label.toLowerCase().replace(/\s+/g, "-");
  return (
    <label htmlFor={inputId} className="block">
      <span className="mb-1 block font-sans text-[10px] font-medium uppercase tracking-widest text-muted">
        {label}
      </span>
      <input
        id={inputId}
        className={`w-full border-0 border-b border-ink bg-transparent py-2 font-mono text-sm text-ink placeholder:text-muted focus:border-accent focus:outline-none ${className}`}
        {...rest}
      />
    </label>
  );
}
