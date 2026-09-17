"""Desktop application launcher.

This module is intentionally small: application construction remains in
``updated_main_app`` for backwards compatibility, while this file provides a
clear, documented entry point for scripts and future packaging.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Create and run the quantized desktop application.

    ``mainloop`` is the CustomTkinter event loop; it owns UI callbacks such as
    tab selection, video analysis, and live-preview refreshes.
    """
    from updated_main_app import LipadQuantizedApp

    # The constructor initializes the application state and builds the shell.
    app = LipadQuantizedApp()
    # mainloop dispatches button callbacks and keeps the desktop UI responsive.
    app.mainloop()


if __name__ == "__main__":
    main()
