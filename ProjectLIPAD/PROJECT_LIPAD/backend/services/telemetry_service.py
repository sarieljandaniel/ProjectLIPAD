from __future__ import annotations

import asyncio
import re
import socket
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Callable

from backend.config import TELEMETRY_PORT


@dataclass
class TelemetryPacket:
    timestamp: str
    source_ip: str
    message: str
    distance_mm: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class TelemetryService:
    """UDP listener for Raspberry Pi hotspot telemetry (port 50007)."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[TelemetryPacket], None]] = []
        self._latest_distance_mm: float | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._history: list[TelemetryPacket] = []
        self._max_history = 200

    @property
    def latest_distance_mm(self) -> float | None:
        return self._latest_distance_mm

    def subscribe(self, callback: Callable[[TelemetryPacket], None]) -> None:
        self._subscribers.append(callback)

    def get_history(self) -> list[dict]:
        return [p.to_dict() for p in self._history]

    def _parse_distance(self, message: str) -> float | None:
        match = re.search(r"DISTANCE:\s*(\d+\.?\d*)", message, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"(\d+\.?\d*)\s*mm", message, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def _emit(self, packet: TelemetryPacket) -> None:
        self._history.append(packet)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        if packet.distance_mm is not None:
            self._latest_distance_mm = packet.distance_mm
        for cb in list(self._subscribers):
            try:
                cb(packet)
            except Exception:
                pass

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._listen_blocking, daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _listen_blocking(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", TELEMETRY_PORT))
            sock.settimeout(1.0)
            while self._running:
                try:
                    data, addr = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break

                message = data.decode("utf-8", errors="replace").strip()
                distance = self._parse_distance(message)
                packet = TelemetryPacket(
                    timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    source_ip=addr[0],
                    message=message,
                    distance_mm=distance,
                )
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._emit, packet)
                else:
                    self._emit(packet)
        finally:
            sock.close()


telemetry_service = TelemetryService()
