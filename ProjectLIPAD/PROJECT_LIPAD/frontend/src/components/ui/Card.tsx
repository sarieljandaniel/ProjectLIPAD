import { ReactNode } from "react";

type Props = {
  children: ReactNode;
  className?: string;
  inverted?: boolean;
};

export function Card({ children, className = "", inverted = false }: Props) {
  const bg = inverted ? "bg-ink text-newsprint border-newsprint" : "bg-newsprint text-ink border-ink";
  return (
    <div className={`border ${bg} ${className}`}>{children}</div>
  );
}
