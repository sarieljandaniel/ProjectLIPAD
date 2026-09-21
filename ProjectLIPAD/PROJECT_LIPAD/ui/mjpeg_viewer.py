"""Low-latency MJPEG camera viewer for the CustomTkinter desktop app.

OpenCV decodes the HTTP multipart stream on a daemon thread and schedules only
UI updates on Tk's main thread, so a stalled camera connection cannot freeze the
application.
"""

from __future__ import annotations

import threading
from typing import Any

import cv2
from PIL import Image

import customtkinter as ctk


def _stop_mjpeg_viewer(app: Any) -> None:
    app._mjpeg_stop_event.set()
    capture = getattr(app, "_mjpeg_capture", None)
    if capture is not None:
        capture.release()
    app._mjpeg_capture = None
    app._mjpeg_running = False


def stop_mjpeg_viewer(app: Any) -> None:
    _stop_mjpeg_viewer(app)
    label = getattr(app, "mjpeg_preview_lbl", None)
    if label is not None and label.winfo_exists():
        label.configure(image=None, text="MJPEG feed stopped")
    button = getattr(app, "mjpeg_toggle_btn", None)
    if button is not None and button.winfo_exists():
        button.configure(text="Start camera feed")


def _publish_frame(app: Any, frame) -> None:
    if not getattr(app, "_mjpeg_running", False):
        return
    label = getattr(app, "mjpeg_preview_lbl", None)
    if label is None or not label.winfo_exists():
        return
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((720, 405), Image.Resampling.LANCZOS)
        ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
        app._mjpeg_imgtk = ctk_image
        label.configure(image=ctk_image, text="")
    except Exception:
        pass


def start_mjpeg_viewer(app: Any) -> None:
    if getattr(app, "_mjpeg_running", False):
        stop_mjpeg_viewer(app)
        return

    url = (app.mjpeg_url.get() or "").strip()
    if not url:
        app._set_mjpeg_status("Enter the Raspberry Pi MJPEG URL first.")
        return

    _stop_mjpeg_viewer(app)
    app._mjpeg_stop_event = threading.Event()
    app._mjpeg_running = True
    if app.mjpeg_preview_lbl.winfo_exists():
        app.mjpeg_preview_lbl.configure(image=None, text="Connecting to camera feed…")
    app.mjpeg_toggle_btn.configure(text="Stop camera feed")
    app.mjpeg_status_lbl.configure(text="Connecting…")

    def _reader() -> None:
        capture = cv2.VideoCapture(url)
        app._mjpeg_capture = capture
        if not capture.isOpened():
            capture.release()
            app._mjpeg_capture = None
            app._mjpeg_running = False
            app.after(0, lambda: app._set_mjpeg_status("Unable to open MJPEG URL"))
            return
        try:
            while not app._mjpeg_stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                app.after(0, lambda f=frame: _publish_frame(app, f))
        finally:
            capture.release()
            app._mjpeg_capture = None
            if not app._mjpeg_stop_event.is_set():
                app._mjpeg_running = False
                app.after(0, lambda: app._set_mjpeg_status("Camera feed disconnected"))

    threading.Thread(target=_reader, daemon=True, name="mjpeg-camera-reader").start()


def configure_mjpeg_controls(app: Any, parent: Any, tokens: Any, body_label, labeled_entry, newsprint_button) -> None:
    """Add MJPEG controls below the existing Inspection Manager grid.

    The inspection page uses ``grid`` on its parent, so this section must also
    use ``grid``. Mixing ``pack`` and ``grid`` on the same parent prevents the
    controls from being created.
    """
    if getattr(app, "mjpeg_controls_frame", None) is not None:
        try:
            if app.mjpeg_controls_frame.winfo_exists():
                return
        except Exception:
            pass

    if not hasattr(app, "mjpeg_url"):
        app.mjpeg_url = ctk.StringVar(value="http://192.168.1.50:5000/video_feed")
    app._mjpeg_stop_event = getattr(app, "_mjpeg_stop_event", threading.Event())
    app._mjpeg_capture = getattr(app, "_mjpeg_capture", None)
    app._mjpeg_running = getattr(app, "_mjpeg_running", False)
    app._mjpeg_imgtk = None

    section = ctk.CTkFrame(parent, fg_color="transparent")
    section.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 12))
    app.mjpeg_controls_frame = section
    body_label(section, tokens, "Direct MJPEG camera feed", mono=True).pack(anchor="w")
    body_label(
        section, tokens,
        "Enter the Raspberry Pi HTTP /video_feed endpoint for display-only viewing.",
        wraplength=620,
    ).pack(anchor="w", pady=(4, 8))
    url_row = ctk.CTkFrame(section, fg_color="transparent")
    url_row.pack(fill="x", pady=(0, 8))
    url_entry, _ = labeled_entry(url_row, tokens, "MJPEG URL", app.mjpeg_url, width=360)
    url_entry.pack(side="left", padx=(0, 8))
    app.mjpeg_toggle_btn = newsprint_button(
        url_row, tokens, "Start camera feed", command=lambda: start_mjpeg_viewer(app), variant="secondary"
    )
    app.mjpeg_toggle_btn.pack(side="left")
    app.mjpeg_status_lbl = body_label(section, tokens, "Camera feed stopped", mono=True)
    app.mjpeg_status_lbl.pack(anchor="w", pady=(0, 8))
    app.mjpeg_preview_lbl = ctk.CTkLabel(
        section, text="Camera feed stopped", text_color=tokens.console_fg,
        fg_color=tokens.console_bg, anchor="center", justify="center", height=260,
    )
    app.mjpeg_preview_lbl.pack(fill="x")


def set_mjpeg_status(app: Any, message: str) -> None:
    label = getattr(app, "mjpeg_status_lbl", None)
    if label is not None and label.winfo_exists():
        label.configure(text=message)
    preview = getattr(app, "mjpeg_preview_lbl", None)
    if preview is not None and preview.winfo_exists() and not getattr(app, "_mjpeg_running", False):
        preview.configure(image=None, text=message)
    button = getattr(app, "mjpeg_toggle_btn", None)
    if button is not None and button.winfo_exists() and not getattr(app, "_mjpeg_running", False):
        button.configure(text="Start camera feed")


def install_status_callback(app: Any) -> None:
    app._set_mjpeg_status = lambda message: set_mjpeg_status(app, message)
