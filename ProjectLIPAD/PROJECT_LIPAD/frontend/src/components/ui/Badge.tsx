type Props = {
  children: string;
  tone?: "default" | "accent" | "inverted";
  className?: string;
};

export function Badge({ children, tone = "default", className = "" }: Props) {
  const tones = {
    default: "border-ink text-ink bg-rule",
    accent: "border-accent text-accent bg-newsprint",
    inverted: "border-newsprint text-newsprint bg-ink",
  };
  return (
    <span
      className={`inline-block border px-2 py-0.5 font-sans text-[10px] font-medium uppercase tracking-widest ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  );
}
