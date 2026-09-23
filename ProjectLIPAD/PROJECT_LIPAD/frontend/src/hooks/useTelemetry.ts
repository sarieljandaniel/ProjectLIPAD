import { useCallback, useEffect, useRef, useState } from "react";
import { TelemetryPacket, api, telemetryWsUrl } from "../api/client";

export function useTelemetry() {
  const [packets, setPackets] = useState<TelemetryPacket[]>([]);
  const [connected, setConnected] = useState(false);
  const [latestDistance, setLatestDistance] = useState<number | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const appendPacket = useCallback((packet: TelemetryPacket) => {
    setPackets((prev) => [...prev.slice(-199), packet]);
    if (packet.distance_mm != null) setLatestDistance(packet.distance_mm);
  }, []);

  useEffect(() => {
    let cancelled = false;

    api.telemetryHistory().then((data) => {
      if (!cancelled && data.packets?.length) {
        setPackets(data.packets.slice(-200));
        const last = [...data.packets].reverse().find((p) => p.distance_mm != null);
        if (last?.distance_mm != null) setLatestDistance(last.distance_mm);
      }
    });

    const connect = () => {
      const ws = new WebSocket(telemetryWsUrl());
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (!cancelled) setTimeout(connect, 2000);
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "packet" && msg.packet) appendPacket(msg.packet);
          if (msg.type === "history" && msg.packets) setPackets(msg.packets.slice(-200));
        } catch {
          /* ignore malformed */
        }
      };
    };

    connect();
    return () => {
      cancelled = true;
      wsRef.current?.close();
    };
  }, [appendPacket]);

  return { packets, connected, latestDistance };
}
