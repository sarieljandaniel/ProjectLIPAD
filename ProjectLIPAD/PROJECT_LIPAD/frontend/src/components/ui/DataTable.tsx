type Props = {
  columns: string[];
  rows: Record<string, string>[];
};

export function DataTable({ columns, rows }: Props) {
  if (!columns.length) {
    return (
      <p className="p-4 font-sans text-sm text-muted">No tabular data available.</p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] border-collapse text-left">
        <thead>
          <tr className="border-b-4 border-ink">
            {columns.map((col) => (
              <th
                key={col}
                className="px-3 py-2 font-sans text-[10px] font-medium uppercase tracking-widest text-muted"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-rule hover:bg-rule/50">
              {columns.map((col) => (
                <td key={col} className="px-3 py-2 font-mono text-xs text-ink">
                  {row[col] ?? ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
