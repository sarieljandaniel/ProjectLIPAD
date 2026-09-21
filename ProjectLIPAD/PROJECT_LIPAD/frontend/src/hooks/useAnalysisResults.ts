import { useCallback, useEffect, useState } from "react";
import { ResultsPayload, RepairGuidance, api } from "../api/client";

export function useAnalysisResults(pollMs = 4000) {
  const [results, setResults] = useState<ResultsPayload | null>(null);
  const [repair, setRepair] = useState<RepairGuidance | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [r, g] = await Promise.all([api.getResults(), api.getRepair()]);
      setResults(r);
      setRepair(g);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load results");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, pollMs);
    return () => clearInterval(id);
  }, [refresh, pollMs]);

  return { results, repair, loading, error, refresh };
}
