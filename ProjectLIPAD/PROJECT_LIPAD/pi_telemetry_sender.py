#!/usr/bin/env python3
"""Example UDP telemetry sender for Raspberry Pi 4B (hotspot client).

Broadcast LiDAR distance to the ground station on port 50007.
Run on the Pi while connected as hotspot AP (typically 10.42.0.1).
"""

import socket
import time

BROADCAST = "10.42.0.255"
PORT = 50007
INTERVAL_SEC = 0.5


def main() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    distance_mm = 1168.4
    try:
        while True:
            message = f"DISTANCE: {distance_mm:.1f}mm"
            sock.sendto(message.encode("utf-8"), (BROADCAST, PORT))
            print(f"Sent -> {BROADCAST}:{PORT}  {message}")
            distance_mm += 1.0
            if distance_mm > 1300:
                distance_mm = 850.0
            time.sleep(INTERVAL_SEC)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
