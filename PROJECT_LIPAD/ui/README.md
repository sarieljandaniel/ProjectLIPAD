# Desktop app organization

The quantized desktop app is launched with:

```bash
python -m ui.app_launcher
```

The supporting modules are organized as follows:

- `ui/app_config.py` — repository-relative paths and runtime defaults.
- `ui/app_launcher.py` — the small, documented desktop entry point.
- `ui/pages/` — page rendering functions.
- `ui/components.py` — reusable widgets.
- `ui/telemetry.py` — Raspberry Pi telemetry transport.
- `ui/pi_link.py` — Raspberry Pi connection dialog and setup.
- `ui/prerequisites.py` — dependency and FFmpeg setup.

`updated_main_app.py` remains the compatibility application module for now.
The comments in the launcher identify where the constructor and Tk event loop
are used, while the existing page modules keep presentation code out of the
launcher.
