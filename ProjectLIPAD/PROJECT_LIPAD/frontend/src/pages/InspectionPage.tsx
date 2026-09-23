import { useCallback, useEffect, useState } from "react";
import { api, AnalysisState } from "../api/client";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Input } from "../components/ui/Input";
import { Badge } from "../components/ui/Badge";

export function InspectionPage() {
  const [videoPath, setVideoPath] = useState("");
  const [inspectionType, setInspectionType] = useState<"Crack" | "Corrosion">("Crack");
  const [corrosionEnv, setCorrosionEnv] = useState<"Wet" | "Dry">("Wet");
  const [gsd, setGsd] = useState("0.5436");
  const [stride, setStride] = useState("1");
  const [infWidth, setInfWidth] = useState("0");
  const [status, setStatus] = useState<AnalysisState | null>(null);

  const refreshStatus = useCallback(async () => {
    const s = await api.analysisStatus();
    setStatus(s);
  }, []);

  useEffect(() => {
    refreshStatus();
    const id = setInterval(refreshStatus, 3000);
    return () => clearInterval(id);
  }, [refreshStatus]);

  const run = async () => {
    if (!videoPath.trim()) return;
    const result = await api.runAnalysis({
      video_path: videoPath.trim(),
      inspection_type: inspectionType,
      corrosion_env: corrosionEnv,
      gsd: parseFloat(gsd) || 0.5436,
      frame_stride: parseInt(stride, 10) || 1,
      inference_width: parseInt(infWidth, 10) || 0,
    });
    setStatus(result);
  };

  const stop = async () => {
    const result = await api.stopAnalysis();
    setStatus(result);
  };

  return (
    <div>
      <header className="mb-6 border-b-4 border-ink pb-4">
        <p className="font-sans text-[10px] uppercase tracking-widest text-muted">Operations</p>
        <h2 className="font-serif text-3xl">Inspection manager</h2>
      </header>

      <div className="grid grid-cols-1 gap-0 lg:grid-cols-12">
        <section className="lg:col-span-8 lg:border-r lg:border-ink lg:pr-6">
          <Card className="mb-6">
            <div className="border-b border-ink p-4">
              <p className="font-sans text-[10px] uppercase tracking-widest text-muted">Target</p>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button
                  variant={inspectionType === "Crack" ? "primary" : "secondary"}
                  onClick={() => setInspectionType("Crack")}
                >
                  Crack detection
                </Button>
                <Button
                  variant={inspectionType === "Corrosion" ? "primary" : "secondary"}
                  onClick={() => setInspectionType("Corrosion")}
                >
                  Corrosion detection
                </Button>
              </div>
            </div>
            {inspectionType === "Corrosion" && (
              <div className="border-b border-ink p-4">
                <p className="font-sans text-[10px] uppercase tracking-widest text-muted">Environment</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button
                    variant={corrosionEnv === "Wet" ? "primary" : "secondary"}
                    onClick={() => setCorrosionEnv("Wet")}
                  >
                    Wet (marine)
                  </Button>
                  <Button
                    variant={corrosionEnv === "Dry" ? "primary" : "secondary"}
                    onClick={() => setCorrosionEnv("Dry")}
                  >
                    Dry (oxidation)
                  </Button>
                </div>
              </div>
            )}
            <div className="grid grid-cols-1 gap-4 p-4 sm:grid-cols-3">
              <Input label="GSD (mm/px)" value={gsd} onChange={(e) => setGsd(e.target.value)} />
              <Input label="Frame stride" value={stride} onChange={(e) => setStride(e.target.value)} />
              <Input
                label="Inference width (0=native)"
                value={infWidth}
                onChange={(e) => setInfWidth(e.target.value)}
              />
            </div>
          </Card>

          <Card>
            <div className="p-4">
              <Input
                label="Video path (full path to MP4)"
                value={videoPath}
                onChange={(e) => setVideoPath(e.target.value)}
                placeholder="C:\path\to\inspection.mp4"
              />
              <p className="mt-2 font-sans text-xs text-muted">
                Paste the absolute path to your inspection video. Browser file pickers are not used for engine compatibility on Windows.
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <Button onClick={run} disabled={status?.status === "running"}>
                  Analyze video
                </Button>
                <Button variant="secondary" onClick={stop}>
                  Stop
                </Button>
              </div>
            </div>
          </Card>
        </section>

        <aside className="mt-6 lg:col-span-4 lg:mt-0 lg:pl-6">
          <Card inverted>
            <div className="p-4">
              <p className="font-sans text-[10px] uppercase tracking-widest text-rule">Engine status</p>
              <div className="mt-3 flex items-center gap-2">
                <Badge tone="inverted">{status?.status ?? "idle"}</Badge>
                {inspectionType === "Corrosion" && <Badge tone="accent">Corrosion</Badge>}
              </div>
              <p className="mt-4 font-mono text-sm leading-relaxed">{status?.message ?? "Ready"}</p>
              {status?.last_annotated_video && (
                <p className="mt-4 break-all font-mono text-[10px] text-rule">
                  Output: {status.last_annotated_video}
                </p>
              )}
            </div>
          </Card>
        </aside>
      </div>
    </div>
  );
}
