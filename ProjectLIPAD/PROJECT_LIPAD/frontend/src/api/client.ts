const API_BASE = "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),
  telemetryStatus: () =>
    request<{ port: number; latest_distance_mm: number | null; packet_count: number }>(
      "/api/telemetry/status",
    ),
  telemetryHistory: () => request<{ packets: TelemetryPacket[] }>("/api/telemetry/history"),
  analysisStatus: () => request<AnalysisState>("/api/analysis/status"),
  runAnalysis: (body: RunAnalysisBody) =>
    request<AnalysisState>("/api/analysis/run", { method: "POST", body: JSON.stringify(body) }),
  stopAnalysis: () => request<AnalysisState>("/api/analysis/stop", { method: "POST" }),
  getResults: () => request<ResultsPayload>("/api/results"),
  getRepair: () => request<RepairGuidance>("/api/results/repair"),
  clearResults: () => request<{ cleared: number }>("/api/results", { method: "DELETE" }),
};

export type TelemetryPacket = {
  timestamp: string;
  source_ip: string;
  message: string;
  distance_mm: number | null;
};

export type AnalysisState = {
  status: string;
  message: string;
  video_path: string | null;
  inspection_type: string;
  corrosion_env: string;
  last_annotated_video: string | null;
};

export type RunAnalysisBody = {
  video_path: string;
  inspection_type: "Crack" | "Corrosion";
  corrosion_env: "Wet" | "Dry";
  gsd: number;
  frame_stride: number;
  inference_width: number;
};

export type ResultsPayload = {
  rows: Record<string, string>[];
  columns: string[];
  path: string | null;
  summary: {
    total?: number;
    cracks?: number;
    corrosion?: number;
    latest_type?: string;
    latest_severity?: string;
  };
};

export type RepairGuidance = {
  headline: string;
  body: string;
  notes: string[];
  severity: string;
  has_crack: boolean;
  has_corrosion: boolean;
};

export function telemetryWsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.host;
  return `${proto}://${host}/api/telemetry/ws`;
}
