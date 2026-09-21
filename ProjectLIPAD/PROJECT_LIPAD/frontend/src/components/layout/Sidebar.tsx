type TabId = "home" | "inspection" | "analysis" | "reports";

type NavItem = { id: TabId; label: string };

const NAV: NavItem[] = [
  { id: "home", label: "Home" },
  { id: "inspection", label: "Inspection Manager" },
  { id: "analysis", label: "Analysis Overview" },
  { id: "reports", label: "Reports" },
];

type Props = {
  active: TabId;
  onSelect: (id: TabId) => void;
};

export function Sidebar({ active, onSelect }: Props) {
  return (
    <aside className="flex w-full flex-col border-b-4 border-ink bg-newsprint md:w-56 md:shrink-0 md:border-b-0 md:border-r-4">
      <div className="border-b border-ink px-5 py-6">
        <p className="font-sans text-[10px] font-medium uppercase tracking-[0.2em] text-muted">
          Structural Health
        </p>
        <h1 className="mt-1 font-serif text-2xl text-ink">LiPAD AI</h1>
      </div>
      <nav className="flex flex-1 flex-col" aria-label="Main navigation">
        {NAV.map((item) => {
          const isActive = active === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelect(item.id)}
              aria-current={isActive ? "page" : undefined}
              className={`border-b border-rule px-5 py-4 text-left font-sans text-xs font-medium uppercase tracking-widest transition-colors min-h-[44px] ${
                isActive ? "bg-ink text-newsprint" : "text-ink hover:bg-rule"
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </nav>
      <div className="border-t border-ink px-5 py-4">
        <p className="font-sans text-[10px] uppercase tracking-widest text-muted">Edition</p>
        <p className="font-mono text-xs text-ink">v1.0 · Newsprint UI</p>
      </div>
    </aside>
  );
}

export type { TabId };
