import { ReactNode } from "react";

type Props = {
  children: ReactNode;
};

export function MainLayout({ children }: Props) {
  return (
    <main className="min-h-0 flex-1 overflow-y-auto bg-newsprint">
      <div className="mx-auto max-w-7xl p-4 sm:p-6 md:p-8">{children}</div>
    </main>
  );
}
