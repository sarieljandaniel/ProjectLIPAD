import { useAnalysisResults } from "../hooks/useAnalysisResults";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { DataTable } from "../components/ui/DataTable";
import { Badge } from "../components/ui/Badge";
import { api } from "../api/client";

export function AnalysisPage() {
  const { results, repair, loading, error, refresh } = useAnalysisResults(5000);

  const clear = async () => {
    await api.clearResults();
    await refresh();
  };

  const summary = results?.summary ?? {};
  const rows = results?.rows ?? [];
  const columns = results?.columns ?? [];

  return (
    <div>
      <header className="mb-6 flex flex-col gap-4 border-b-4 border-ink pb-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="font-sans text-[10px] uppercase tracking-widest text-muted">Analytics</p>
          <h2 className="font-serif text-3xl">Analysis overview</h2>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" onClick={refresh}>
            Refresh
          </Button>
          <Button variant="secondary" onClick={clear}>
            Clear results
          </Button>
        </div>
      </header>

      <div className="mb-6 grid grid-cols-2 gap-0 border border-ink sm:grid-cols-4">
        {[
          { label: "Total defects", value: summary.total ?? 0 },
          { label: "Cracks", value: summary.cracks ?? 0 },
          { label: "Corrosion", value: summary.corrosion ?? 0 },
          { label: "Latest severity", value: summary.latest_severity ?? "—" },
        ].map((item, i) => (
          <Card
            key={item.label}
            className={`border-0 ${i < 3 ? "border-b border-ink sm:border-b-0 sm:border-r" : ""}`}
          >
            <div className="p-4">
              <p className="font-sans text-[10px] uppercase tracking-widest text-muted">{item.label}</p>
              <p className="mt-2 font-mono text-2xl tabular-nums">{item.value}</p>
            </div>
          </Card>
        ))}
      </div>

      {loading && <p className="font-sans text-sm text-muted">Loading results…</p>}
      {error && (
        <Card className="mb-4 border-accent p-4">
          <p className="font-sans text-sm text-accent">{error}</p>
        </Card>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <section className="lg:col-span-8">
          <Card>
            <div className="flex items-center justify-between border-b border-ink px-4 py-3">
              <h3 className="font-serif text-lg">Defect analytics</h3>
              {summary.latest_type && <Badge tone="accent">{summary.latest_type}</Badge>}
            </div>
            {rows.length === 0 ? (
              <p className="p-6 font-sans text-sm text-muted">
                No defects detected yet. Run a corrosion or crack analysis in Inspection Manager.
              </p>
            ) : (
              <DataTable columns={columns} rows={rows} />
            )}
          </Card>
        </section>

        <aside className="lg:col-span-4">
          <Card inverted>
            <div className="border-b border-newsprint p-4">
              <p className="font-sans text-[10px] uppercase tracking-widest text-rule">Repair guidance</p>
              <h3 className="mt-2 font-serif text-xl">{repair?.headline ?? "Awaiting data"}</h3>
            </div>
            <div className="p-4">
              <p className="font-sans text-sm leading-relaxed">{repair?.body}</p>
              {repair?.notes?.map((note) => (
                <p key={note} className="mt-3 font-sans text-xs text-rule">
                  {note}
                </p>
              ))}
            </div>
          </Card>
        </aside>
      </div>
    </div>
  );
}
