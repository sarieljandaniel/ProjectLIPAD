"""UDP telemetry listener for Raspberry Pi hotspot (port 50007)."""

from __future__ import annotations

import re
import socket
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

TELEMETRY_PORT = 50007


@dataclass
class TelemetryPacket:
    timestamp: str
    source_ip: str
    message: str
    distance_mm: float | None = None


class TelemetryListener:
    def __init__(self, on_packet: Callable[[TelemetryPacket], None] | None = None) -> None:
        self._on_packet = on_packet
        self._running = False
        self._thread: threading.Thread | None = None
        self.latest_distance_mm: float | None = None
        self.connected = False
        self.packet_count = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _parse_distance(self, message: str) -> float | None:
        match = re.search(r"DISTANCE:\s*(\d+\.?\d*)", message, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"(\d+\.?\d*)\s*mm", message, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def _listen(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", TELEMETRY_PORT))
            sock.settimeout(1.0)
            self.connected = True
            while self._running:
                try:
                    data, addr = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                message = data.decode("utf-8", errors="replace").strip()
                distance = self._parse_distance(message)
                if distance is not None:
                    self.latest_distance_mm = distance
                self.packet_count += 1
                packet = TelemetryPacket(
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    source_ip=addr[0],
                    message=message,
                    distance_mm=distance,
                )
                if self._on_packet:
                    self._on_packet(packet)
        except OSError as exc:
            if self._on_packet:
                err = TelemetryPacket(
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    source_ip="local",
                    message=f"[ERROR] UDP bind failed on port {TELEMETRY_PORT}: {exc}",
                    distance_mm=None,
                )
                self._on_packet(err)
        finally:
            self.connected = False
            sock.close()
