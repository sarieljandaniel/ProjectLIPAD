from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.telemetry_service import telemetry_service

router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


@router.get("/status")
async def telemetry_status():
    return {
        "port": 50007,
        "latest_distance_mm": telemetry_service.latest_distance_mm,
        "packet_count": len(telemetry_service.get_history()),
    }


@router.get("/history")
async def telemetry_history():
    return {"packets": telemetry_service.get_history()}


@router.websocket("/ws")
async def telemetry_ws(websocket: WebSocket):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()

    def on_packet(packet):
        try:
            queue.put_nowait(packet.to_dict())
        except Exception:
            pass

    telemetry_service.subscribe(on_packet)
    await websocket.send_json({"type": "history", "packets": telemetry_service.get_history()})

    try:
        while True:
            try:
                packet = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json({"type": "packet", "packet": packet})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
