#!/usr/bin/env python3
"""Launch Project LiPAD API server (serves built frontend when dist/ exists)."""

import os
import sys
import webbrowser
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIST = ROOT / "frontend" / "dist"


def main() -> None:
    host = os.environ.get("LIPAD_HOST", "127.0.0.1")
    port = int(os.environ.get("LIPAD_PORT", "8000"))
    url = f"http://{host}:{port}"

    if DIST.exists():
        print(f"[LiPAD] Serving dashboard at {url}")
    else:
        print(f"[LiPAD] API at {url} — build frontend: cd frontend && npm run build")
        print("[LiPAD] Dev UI: cd frontend && npm run dev (proxies API)")

    if DIST.exists() and os.environ.get("LIPAD_NO_BROWSER") != "1":
        webbrowser.open(url)

    uvicorn.run("backend.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
