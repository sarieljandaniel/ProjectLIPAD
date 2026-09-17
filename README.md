# Project LiPAD

Project LiPAD is a structural-health inspection system for analyzing infrastructure imagery and video. The repository contains the desktop inspection application, quantized YOLO inference engines, Raspberry Pi camera/telemetry helpers, a web API and frontend, and model-training material.

This document focuses on the **current desktop application**, its processing lifecycle, repository navigation, setup, and operating instructions.

## Contents

- [System overview](#system-overview)
- [Desktop application](#desktop-application)
- [Application startup](#application-startup)
- [Desktop processing workflows](#desktop-processing-workflows)
- [Navigating the repository](#navigating-the-repository)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the desktop application](#running-the-desktop-application)
- [Using the desktop application](#using-the-desktop-application)
- [Outputs and generated files](#outputs-and-generated-files)
- [Raspberry Pi live workflow](#raspberry-pi-live-workflow)
- [Troubleshooting](#troubleshooting)
- [Development notes](#development-notes)

## System overview

Project LiPAD has several related subsystems:

1. **Desktop application** — a CustomTkinter interface used to select inspections, analyze MP4 files, connect to a Raspberry Pi camera, monitor telemetry, and inspect/export results.
2. **Quantized inference engine** — a separate Python process that performs video or live-stream inference and writes CSV/video/preview outputs.
3. **Raspberry Pi integration** — SSH-based camera control, network video streaming, and telemetry reception.
4. **Web application** — a separate FastAPI backend and frontend launched by `run_lipad.py`.
5. **Training and notebook material** — notebooks and scripts used to train or evaluate models; these are not required for normal desktop operation.

The desktop app does not perform the long-running inference work directly in the Tkinter event loop. It launches the inference engine with `subprocess.Popen`, monitors the process from a background thread, and updates the UI through the Tk event queue. This keeps the interface responsive while analysis is running.

## Desktop application

The primary desktop module is:

```text
PROJECT_LIPAD/updated_main_app.py
```

The main class is `LipadQuantizedApp`. It is a CustomTkinter window with four principal areas:

- **Home** — application status, telemetry summary, connection state, and live status.
- **Inspection Manager** — inspection selection, video management, engine tuning, Raspberry Pi settings, firewall access, and live controls.
- **Analysis Overview** — result summaries and live/inference previews.
- **Reports** — CSV viewing/export actions and report status.

The interface is assembled from reusable UI modules under `PROJECT_LIPAD/ui/`. The page modules render page content, while the application class coordinates state, subprocesses, files, and callbacks.

### Desktop responsibilities

`LipadQuantizedApp` currently handles:

- CustomTkinter window creation and theme switching.
- Navigation between application pages.
- MP4 selection, selection changes, removal, and opening the containing folder.
- Parsing inference settings such as GSD, frame stride, and inference width.
- Building command-line arguments for the quantized engine.
- Starting and stopping offline video analysis.
- Starting and stopping live analysis.
- Creating and clearing live preview images.
- Connecting to the Raspberry Pi through Paramiko/SSH.
- Requesting a Windows Firewall rule for the live-stream port.
- Receiving telemetry packets and updating the Home page.
- Exporting results to a user-selected CSV file.
- Cleaning up child processes and telemetry on application exit.

## Application startup

There are two supported desktop launch styles.

### Recommended module launcher

From the `PROJECT_LIPAD` directory:

```bash
python -m ui.app_launcher
```

`ui.app_launcher` imports `LipadQuantizedApp`, constructs the window, and calls CustomTkinter's `mainloop()`. The event loop dispatches button callbacks, page navigation, telemetry updates, and preview refreshes.

### Compatibility launcher

From the `PROJECT_LIPAD` directory:

```bash
python updated_main_app.py
```

This is the original application entry point and remains available for compatibility.

From the repository root, you can also run:

```bash
python PROJECT_LIPAD/launch_desktop.py
```

### Startup sequence

At startup, the desktop application:

1. Adds the application directory to Python's import path.
2. Makes local FFmpeg available when the prerequisite helper supports it.
3. Optionally checks/bootstrap-installs prerequisites when `updated_main_app.py` is run directly.
4. Loads CustomTkinter, Pillow, pandas, and the UI modules.
5. Creates the `LipadQuantizedApp` window and initializes application state.
6. Starts the telemetry listener thread.
7. Builds the sidebar and default Home page.
8. Enters the Tk event loop.

## Desktop processing workflows

### Offline MP4 analysis

The offline workflow is:

1. Open **Inspection Manager**.
2. Choose an MP4 file with the upload control.
3. Select the inspection type and, for corrosion, the environment.
4. Adjust optional inference settings:
   - GSD in millimeters per pixel.
   - Frame stride.
   - Inference width; `0` uses the engine default behavior.
5. Start analysis.
6. The desktop app creates an annotated-video destination under `data/annotated/`.
7. It starts `Project_LIPAD_AI/lipad_runtime_engine_quantized.py` as a child process.
8. The engine writes its stdout/stderr to `data/engine_error.log`.
9. On success, the engine writes CSV results and an annotated MP4.
10. The UI updates the status and makes the results available to Analysis Overview and Reports.

The desktop process and engine process are separate. Use **Stop Analysis** to terminate the child engine rather than closing the terminal or killing Python manually.

### Live Raspberry Pi analysis

The live workflow is:

1. Configure the PC address, listen host, port, protocol, resolution, bitrate, and Raspberry Pi SSH settings.
2. Allow the configured port through Windows Firewall if needed.
3. Start the live engine from Inspection Manager.
4. The desktop application validates the selected configuration and opens the Pi-link dialog.
5. The quantized engine starts in live mode and listens for the Pi stream.
6. After the listener is ready, the desktop app can start `rpicam-vid` on the Pi through SSH.
7. The Pi sends an MPEG-TS stream over TCP or UDP to the configured PC address and port.
8. The engine writes raw and inference preview JPEGs into `data/`.
9. The UI polls those files and displays the live previews.
10. Telemetry packets are received independently by `TelemetryListener` and shown on the Home page.
11. Stopping the workflow terminates the engine, closes the SSH connection, stops `rpicam-vid`, and removes stale preview files.

The default live settings currently include:

| Setting | Default |
|---|---:|
| Listen host | `0.0.0.0` |
| Port | `5000` |
| Protocol | `tcp` |
| Width | `1280` |
| Height | `720` |
| Bitrate | `3000000` |
| Pi host | `lipad.local` |
| Pi user | `lipad` |

Change these values when the local network or Raspberry Pi configuration differs.

### Telemetry processing

`ui.telemetry.TelemetryListener` runs separately from the UI thread. Incoming packets are queued back onto the Tk event loop through `after(0, ...)`. The application stores a bounded recent packet history, updates the current LiDAR distance, appends readable lines to the telemetry log, and refreshes Home-page status values.

## Navigating the repository

The important directories and files are:

```text
ProjectLIPAD/
├── PROJECT_LIPAD/
│   ├── updated_main_app.py              # Main quantized desktop application
│   ├── launch_desktop.py                # Repository-root-compatible desktop launcher
│   ├── run_lipad.py                     # FastAPI/web frontend launcher; not the desktop app
│   ├── requirements.txt                 # Python dependencies
│   ├── Project_LIPAD_AI/
│   │   ├── lipad_runtime_engine_quantized.py  # Engine used by updated_main_app.py
│   │   └── lipad_runtime_engine.py            # Other/legacy runtime engine
│   ├── models/                          # Model files, if included locally
│   ├── data/                            # Runtime outputs, logs, previews, and CSV files
│   ├── ui/
│   │   ├── app_launcher.py              # Recommended desktop module launcher
│   │   ├── app_config.py                # Centralized path/default configuration
│   │   ├── components.py                # Reusable widgets and labels
│   │   ├── pages/
│   │   │   ├── home.py                  # Home page renderer
│   │   │   ├── inspection.py             # Inspection Manager renderer
│   │   │   ├── analysis.py               # Analysis Overview renderer
│   │   │   └── reports.py                # Reports renderer
│   │   │   ├── pi_link.py                # Raspberry Pi connection dialog
│   │   │   ├── prerequisites.py          # Dependency and FFmpeg setup
│   │   │   ├── sidebar.py                # Sidebar navigation
│   │   │   ├── telemetry.py              # Telemetry listener and packet model
│   │   │   ├── theme.py                  # Theme modes and design tokens
│   │   │   └── README.md                 # UI package notes
│   │   ├── backend/                      # FastAPI backend package
│   │   └── frontend/                     # Frontend source/build files
│   ├── lipad_receiver.ps1                # PowerShell receiving/helper script
│   └── pi_telemetry_sender.py            # Raspberry Pi telemetry sender
├── LIPAD_YOLO_TRAINING/                  # Training assets and scripts
└── Project_LiPAD.ipynb                   # Notebook-based experimentation/training
```

### Which file should I edit?

- Change desktop behavior: `PROJECT_LIPAD/updated_main_app.py`.
- Change page layout: the relevant file under `PROJECT_LIPAD/ui/pages/`.
- Change reusable controls: `PROJECT_LIPAD/ui/components.py`.
- Change colors/theme: `PROJECT_LIPAD/ui/theme.py`.
- Change telemetry parsing/listening: `PROJECT_LIPAD/ui/telemetry.py`.
- Change Pi connection instructions/dialog behavior: `PROJECT_LIPAD/ui/pi_link.py`.
- Change dependency checks or FFmpeg setup: `PROJECT_LIPAD/ui/prerequisites.py`.
- Change inference behavior: `PROJECT_LIPAD/Project_LIPAD_AI/lipad_runtime_engine_quantized.py`.
- Change the web API: `PROJECT_LIPAD/backend/`.
- Change the web UI: `PROJECT_LIPAD/frontend/`.
- Change model training: `LIPAD_YOLO_TRAINING/` or `Project_LiPAD.ipynb`.

`main_app.py` and `main_app_quantized.py` are older desktop variants. Use `updated_main_app.py` for the current quantized desktop workflow unless a specific legacy behavior is required.

## Requirements

The desktop application is Python-based and currently uses these categories of dependencies:

- Python 3.10 or newer is recommended.
- CustomTkinter for the desktop interface.
- pandas for CSV loading/export.
- Pillow for preview images.
- OpenCV and the inference stack used by the runtime engine.
- Paramiko for optional Raspberry Pi SSH control.
- FFmpeg for media handling where required by the engine or live workflow.

Install the exact repository dependencies from the application directory:

```bash
cd PROJECT_LIPAD
python -m pip install -r requirements.txt
```

If a dependency is optional for your workflow, the application prerequisite screen can report or install supported prerequisites. Review the prompt carefully before allowing installation.

## Installation

### Windows PowerShell

```powershell
git clone https://github.com/sarieljandaniel/ProjectLIPAD.git
cd ProjectLIPAD
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r PROJECT_LIPAD\requirements.txt
```

### Windows Command Prompt

```bat
git clone https://github.com/sarieljandaniel/ProjectLIPAD.git
cd ProjectLIPAD
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r PROJECT_LIPAD\requirements.txt
```

### Linux/macOS

```bash
git clone https://github.com/sarieljandaniel/ProjectLIPAD.git
cd ProjectLIPAD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r PROJECT_LIPAD/requirements.txt
```

The live Pi workflow is primarily designed around a Windows desktop because it can request Windows Firewall access and uses Windows-oriented defaults. Offline analysis may work on other operating systems if all engine dependencies and paths are available.

## Running the desktop application

From the repository root:

```bash
python PROJECT_LIPAD/launch_desktop.py
```

Or from inside `PROJECT_LIPAD`:

```bash
python -m ui.app_launcher
```

For the original direct entry point:

```bash
cd PROJECT_LIPAD
python updated_main_app.py
```

To open the web/API application instead, use:

```bash
cd PROJECT_LIPAD
python run_lipad.py
```

`run_lipad.py` is not the desktop application; it starts Uvicorn and the FastAPI backend.

## Using the desktop application

### 1. Home

Use Home to confirm that the application is responsive and to monitor:

- Desktop status.
- Raspberry Pi telemetry connection state.
- Latest telemetry packet.
- Current LiDAR distance.
- Analysis/streaming state.

### 2. Inspection Manager

Use Inspection Manager to:

- Select `Crack` or another supported inspection target.
- Select `Wet` or `Dry` for corrosion inspections.
- Add one or more MP4 files.
- Select or remove a queued video.
- Configure engine tuning values.
- Configure live-stream and Raspberry Pi connection settings.
- Copy the generated `rpicam-vid` command.
- Request firewall access.
- Start or stop offline/live analysis.

Do not remove a video while the analysis engine is running.

### 3. Analysis Overview

Use Analysis Overview to review output data and live/inference previews. If live analysis is active, previews are refreshed periodically from the JPEG files written by the runtime engine.

### 4. Reports

Use Reports to export the generated results as a CSV file. The application prefers `MorphologicalResults.csv` and falls back to `results.csv` when necessary.

## Outputs and generated files

Runtime files are normally written below:

```text
PROJECT_LIPAD/data/
├── MorphologicalResults.csv       # Detailed morphological engine output
├── results.csv                    # UI-compatible result output
├── engine_error.log               # Child-engine stdout/stderr
├── live_preview.jpg               # Live inference preview
├── live_raw_preview.jpg           # Raw live-stream preview
├── live_ready.flag                # Engine readiness marker
└── annotated/
    └── <video>_annotated.mp4      # Offline annotated video
```

The `data/` directory may contain generated files from previous runs. Delete or use the application's clear-results action only when you no longer need those outputs.

## Raspberry Pi live workflow

Before starting live analysis, verify:

- The PC and Pi are on the same LAN or routed network.
- The Pi hostname/IP is correct.
- SSH is enabled on the Pi.
- The configured SSH user can run `rpicam-vid`.
- The PC's selected IP is reachable from the Pi.
- The selected TCP/UDP port is allowed through the firewall.
- The camera is connected and supported by the Pi configuration.
- `paramiko` is installed in the active Python environment.

The generated camera command contains the selected resolution, bitrate, protocol, PC address, and port. If the Pi cannot connect, copy the command and test it manually from an SSH session to isolate camera, network, or application issues.

## Troubleshooting

### The desktop window does not open

- Activate the virtual environment.
- Confirm that CustomTkinter and Pillow are installed.
- Run `python -m ui.app_launcher` from inside `PROJECT_LIPAD` so the `ui` package is importable.
- On Linux, confirm that a graphical display is available.

### The model cannot be found

The current desktop module still contains a historical machine-specific default model path. Pass a valid model path when constructing the app or update the default in `updated_main_app.py` for your environment. The model must exist before a crack analysis can start.

### Analysis fails

Inspect:

```text
PROJECT_LIPAD/data/engine_error.log
```

Also confirm that the selected video is a readable MP4, the runtime engine exists, and the required model/dependencies are installed.

### Live preview stays blank

- Confirm that the live engine is running.
- Check `data/engine_error.log`.
- Verify the Pi address, PC address, protocol, and port.
- Check the firewall rule.
- Confirm that the Pi is sending to the same address and port shown by the generated command.
- Remove stale preview files and restart the live workflow.

### SSH cannot start the camera

- Confirm the Pi host resolves, for example with `ping lipad.local`.
- Confirm SSH credentials.
- Confirm `paramiko` is installed.
- Test an SSH connection outside the application.
- Confirm `rpicam-vid` is installed and the camera is available on the Pi.

### Port access fails

Use the application's firewall-access action on Windows, or create a narrowly scoped inbound rule manually. The live workflow cannot receive a stream if the selected port is blocked.

## Development notes

- Keep long-running work out of the Tk event loop. Use a worker thread or child process and marshal UI changes back with `after(...)`.
- The engine is intentionally executed as a separate process so it can be stopped independently and so model inference does not freeze the interface.
- Keep generated files in `PROJECT_LIPAD/data/`; do not commit logs, previews, or user videos.
- Add page-specific UI to `ui/pages/` instead of placing layout code in the launcher.
- Add reusable widgets to `ui/components.py`.
- Update this README when launch commands, output paths, or the desktop workflow changes.
- The `ui/app_config.py` module provides a central place for repository-relative paths and defaults as the application is further modularized.

## License and project status

Refer to the repository's license and project metadata for distribution terms. This README documents the current working structure and may need updates as the desktop application and web stack continue to evolve.
