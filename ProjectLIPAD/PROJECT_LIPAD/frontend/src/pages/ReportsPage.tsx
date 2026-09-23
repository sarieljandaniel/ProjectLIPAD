import { useState } from "react";
import { api } from "../api/client";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";

export function ReportsPage() {
  const [message, setMessage] = useState("");

  const exportHint = async () => {
    try {
      const data = await api.getResults();
      if (!data.rows?.length) {
        setMessage("No report available yet. Run an analysis first.");
        return;
      }
      const header = data.columns.join(",");
      const body = data.rows
        .map((row) => data.columns.map((c) => JSON.stringify(row[c] ?? "")).join(","))
        .join("\n");
      const blob = new Blob([`${header}\n${body}`], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "LiPAD_Report.csv";
      a.click();
      URL.revokeObjectURL(url);
      setMessage("Report downloaded.");
    } catch (e) {
      setMessage(e instanceof Error ? e.message : "Export failed");
    }
  };

  return (
    <div className="max-w-2xl">
      <header className="mb-6 border-b-4 border-ink pb-4">
        <p className="font-sans text-[10px] uppercase tracking-widest text-muted">Export</p>
        <h2 className="font-serif text-3xl">Reports</h2>
        <p className="drop-cap mt-4 font-serif text-base leading-relaxed">
          Download the latest morphological results as a CSV for archival or third-party review.
        </p>
      </header>
      <Card>
        <div className="p-6">
          <Button onClick={exportHint}>Export CSV</Button>
          {message && <p className="mt-4 font-mono text-sm text-muted">{message}</p>}
        </div>
      </Card>
    </div>
  );
}
