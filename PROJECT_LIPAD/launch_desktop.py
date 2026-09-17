"""Backward-compatible desktop application entry point."""

from __future__ import annotations

from ui.app_launcher import main


if __name__ == "__main__":
    # Delegate to the documented launcher so both invocation styles work.
    main()
