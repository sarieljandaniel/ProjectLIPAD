"""Optional direct MJPEG-enabled desktop entry point.

This keeps the existing quantized live-analysis transport unchanged while adding
the requested HTTP MJPEG display path. Run this module instead of the original
entry point when using a Raspberry Pi endpoint such as /video_feed.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from updated_main_app import LipadQuantizedApp
from ui.mjpeg_viewer import configure_mjpeg_controls, install_status_callback, stop_mjpeg_viewer


class MjpegLipadApp(LipadQuantizedApp):
    """LiPAD desktop application with a threaded HTTP MJPEG viewer."""

    def __init__(self, *args, **kwargs):
        self._mjpeg_controls_added = False
        super().__init__(*args, **kwargs)
        install_status_callback(self)

    def select_tab(self, name: str) -> None:
        if name != "inspection_manager" and getattr(self, "_mjpeg_running", False):
            stop_mjpeg_viewer(self)
        super().select_tab(name)
        if name == "inspection_manager":
            content = self.container.winfo_children()[0] if self.container.winfo_children() else None
            if content is not None:
                configure_mjpeg_controls(
                    self,
                    content,
                    self.tokens,
                    body_label=self._mjpeg_body_label,
                    labeled_entry=self._mjpeg_labeled_entry,
                    newsprint_button=self._mjpeg_button,
                )

    @staticmethod
    def _mjpeg_body_label(parent, tokens, text, **kwargs):
        from ui.components import body_label
        return body_label(parent, tokens, text, **kwargs)

    @staticmethod
    def _mjpeg_labeled_entry(parent, tokens, label, variable, **kwargs):
        from ui.components import labeled_entry
        return labeled_entry(parent, tokens, label, variable, **kwargs)

    @staticmethod
    def _mjpeg_button(parent, tokens, text, **kwargs):
        from ui.components import newsprint_button
        return newsprint_button(parent, tokens, text, **kwargs)

    def destroy(self) -> None:
        if getattr(self, "_mjpeg_running", False):
            stop_mjpeg_viewer(self)
        super().destroy()


if __name__ == "__main__":
    app = MjpegLipadApp()
    app.mainloop()
