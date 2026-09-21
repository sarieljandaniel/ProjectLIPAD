import { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary";

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
};

export function Button({ variant = "primary", className = "", children, ...rest }: Props) {
  const base =
    "inline-flex items-center justify-center px-4 py-2.5 font-sans text-xs font-medium uppercase tracking-widest border border-ink transition-colors min-h-[44px] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink disabled:opacity-40";
  const styles =
    variant === "primary"
      ? "bg-ink text-newsprint hover:bg-muted"
      : "bg-transparent text-ink hover:bg-rule";

  return (
    <button className={`${base} ${styles} ${className}`} {...rest}>
      {children}
    </button>
  );
}
