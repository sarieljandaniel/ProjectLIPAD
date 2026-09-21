import { useEffect, useRef } from "react";
import { TelemetryPacket } from "../../api/client";
import { Card } from "../ui/Card";

type Props = {
  packets: TelemetryPacket[];
  connected: boolean;
};

export function TelemetryConsole({ packets, connected }: Props) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [packets]);

  return (
    <Card className="overflow-hidden">
      <div className="flex items-center justify-between border-b border-ink bg-ink px-4 py-3 text-newsprint">
        <p className="font-sans text-[10px] font-medium uppercase tracking-widest">
          Ground station telemetry
        </p>
        <span className="font-mono text-[10px] uppercase">
          {connected ? "Receiving" : "Standby"}
        </span>
      </div>
      <div
        className="h-64 overflow-y-auto bg-ink p-4 font-mono text-xs leading-relaxed text-newsprint"
        role="log"
        aria-live="polite"
        aria-relevant="additions"
      >
        {packets.length === 0 ? (
          <p className="text-muted">
            Waiting for UDP packets from Raspberry Pi 4B on hotspot subnet 10.42.0.x:50007…
          </p>
        ) : (
          packets.map((p, i) => (
            <div
              key={`${p.timestamp}-${i}`}
              className="mb-2 border-b border-rule/20 pb-2 last:border-0 animate-[fadeIn_0.35s_ease-out]"
            >
              <span className="text-muted">[{formatTime(p.timestamp)}]</span>{" "}
              <span className="text-accent">FROM {p.source_ip}</span>
              <span className="text-muted"> → </span>
              <span>{p.message}</span>
              {p.distance_mm != null && (
                <span className="ml-2 text-accent">{p.distance_mm} mm</span>
              )}
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>
    </Card>
  );
}

function formatTime(iso: string) {
  try {
    return new Date(iso).toLocaleTimeString();
  } catch {
    return iso;
  }
}
