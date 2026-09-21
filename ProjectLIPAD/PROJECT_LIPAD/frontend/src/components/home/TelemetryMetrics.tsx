import { TelemetryPacket } from "../../api/client";
import { Badge } from "../ui/Badge";
import { Card } from "../ui/Card";

type Props = {
  connected: boolean;
  latestDistance: number | null;
  packets: TelemetryPacket[];
};

function formatTime(iso: string) {
  try {
    return new Date(iso).toLocaleTimeString();
  } catch {
    return iso;
  }
}

export function TelemetryMetrics({ connected, latestDistance, packets }: Props) {
  const last = packets[packets.length - 1];

  return (
    <div className="grid grid-cols-1 gap-0 border border-ink sm:grid-cols-3">
      <Card className="border-0 border-b border-ink sm:border-b-0 sm:border-r">
        <div className="p-4">
          <p className="font-sans text-[10px] uppercase tracking-widest text-muted">Link status</p>
          <div className="mt-2 flex items-center gap-2">
            <span
              className={`inline-block h-2 w-2 ${connected ? "bg-accent animate-pulse" : "bg-muted"}`}
              aria-hidden
            />
            <p className="font-mono text-lg">{connected ? "Live" : "Reconnecting"}</p>
          </div>
          <Badge tone={connected ? "accent" : "default"} className="mt-3">
            UDP · 50007
          </Badge>
        </div>
      </Card>
      <Card className="border-0 border-b border-ink sm:border-b-0 sm:border-r">
        <div className="p-4">
          <p className="font-sans text-[10px] uppercase tracking-widest text-muted">LiDAR distance</p>
          <p className="mt-2 font-mono text-3xl tabular-nums">
            {latestDistance != null ? `${latestDistance.toFixed(1)}` : "—"}
            <span className="ml-1 text-sm text-muted">mm</span>
          </p>
        </div>
      </Card>
      <Card className="border-0">
        <div className="p-4">
          <p className="font-sans text-[10px] uppercase tracking-widest text-muted">Last packet</p>
          <p className="mt-2 font-mono text-xs leading-relaxed">
            {last ? `${last.source_ip} · ${formatTime(last.timestamp)}` : "Awaiting Pi hotspot telemetry"}
          </p>
        </div>
      </Card>
    </div>
  );
}
