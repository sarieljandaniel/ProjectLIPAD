"""Install Python packages and FFmpeg after the user grants permission."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable

# import name -> pip package
PIP_PACKAGES: tuple[tuple[str, str], ...] = (
    ("customtkinter", "customtkinter"),
    ("pandas", "pandas"),
    ("PIL", "pillow"),
    ("paramiko", "paramiko"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("ultralytics", "ultralytics"),
    ("onnxruntime", "onnxruntime"),
)

GUI_IMPORTS = ("customtkinter", "pandas", "PIL")

FFMPEG_ZIP_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

StatusFn = Callable[[str], None]


def repo_root_from(path: str | None = None) -> str:
    if path:
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def tools_ffmpeg_dir(root: str) -> str:
    return os.path.join(root, "tools", "ffmpeg")


def local_ffmpeg_exe(root: str) -> str | None:
    bin_dir = os.path.join(tools_ffmpeg_dir(root), "bin")
    for name in ("ffmpeg.exe", "ffmpeg"):
        candidate = os.path.join(bin_dir, name)
        if os.path.isfile(candidate):
            return candidate
        nested = os.path.join(tools_ffmpeg_dir(root), name)
        if os.path.isfile(nested):
            return nested
    return None


def ensure_local_ffmpeg_on_path(root: str) -> str | None:
    """Prefer a bundled FFmpeg so the live receiver can find it without a system install."""
    exe = local_ffmpeg_exe(root)
    which = shutil.which("ffmpeg")
    if exe:
        bin_dir = os.path.dirname(exe)
        path = os.environ.get("PATH", "")
        if bin_dir.lower() not in path.lower().split(os.pathsep):
            os.environ["PATH"] = bin_dir + os.pathsep + path
        return exe
    return which


def missing_pip_packages() -> list[str]:
    missing: list[str] = []
    for import_name, pip_name in PIP_PACKAGES:
        if importlib.util.find_spec(import_name) is None:
            missing.append(pip_name)
    return missing


def ffmpeg_available(root: str) -> bool:
    return bool(shutil.which("ffmpeg") or local_ffmpeg_exe(root))


def summarize_prerequisites(root: str) -> str:
    missing = missing_pip_packages()
    ffmpeg_ok = ffmpeg_available(root)
    parts: list[str] = []
    if not missing and ffmpeg_ok:
        return "All required libraries and FFmpeg are installed."
    if missing:
        parts.append("Python packages: " + ", ".join(missing))
    if not ffmpeg_ok:
        parts.append("FFmpeg is not on PATH (needed for the live MPEG-TS receiver).")
    return "Missing — " + " ".join(parts)


def _pip_install(packages: list[str], on_status: StatusFn | None = None) -> None:
    if not packages:
        return
    if on_status:
        on_status(f"Installing Python packages: {', '.join(packages)}…")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        cmd_user = [sys.executable, "-m", "pip", "install", "--user", "--upgrade", *packages]
        result = subprocess.run(cmd_user, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "pip failed").strip().splitlines()
        tail = err[-8:] if err else ["pip failed"]
        raise RuntimeError("\n".join(tail))


def _download_file(url: str, dest: str, on_status: StatusFn | None = None) -> None:
    if on_status:
        on_status("Downloading FFmpeg…")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Project-LiPAD/1.0 (library installer)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        last_pct = -1
        with open(dest, "wb") as fh:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if total > 0 and on_status is not None:
                    pct = min(100, int(downloaded * 100 / total))
                    if pct != last_pct and pct % 5 == 0:
                        last_pct = pct
                        on_status(f"Downloading FFmpeg… {pct}%")


def _install_ffmpeg(root: str, on_status: StatusFn | None = None) -> str:
    existing = ensure_local_ffmpeg_on_path(root)
    if existing:
        return existing

    dest_root = tools_ffmpeg_dir(root)
    os.makedirs(dest_root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lipad-ffmpeg-") as tmp:
        zip_path = os.path.join(tmp, "ffmpeg.zip")
        _download_file(FFMPEG_ZIP_URL, zip_path, on_status)
        if on_status:
            on_status("Extracting FFmpeg…")
        extract_dir = os.path.join(tmp, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        ffmpeg_exe = None
        for dirpath, _dirnames, filenames in os.walk(extract_dir):
            for name in filenames:
                if name.lower() in {"ffmpeg.exe", "ffmpeg"}:
                    ffmpeg_exe = os.path.join(dirpath, name)
                    break
            if ffmpeg_exe:
                break
        if not ffmpeg_exe:
            raise RuntimeError("The FFmpeg archive did not contain ffmpeg.exe.")
        bin_src = os.path.dirname(ffmpeg_exe)
        bin_dst = os.path.join(dest_root, "bin")
        if os.path.isdir(bin_dst):
            shutil.rmtree(bin_dst)
        shutil.copytree(bin_src, bin_dst)

    exe = ensure_local_ffmpeg_on_path(root)
    if not exe:
        raise RuntimeError("FFmpeg was extracted but could not be found.")
    return exe


def install_prerequisites(root: str, on_status: StatusFn | None = None) -> str:
    missing = missing_pip_packages()
    if missing:
        _pip_install(missing, on_status)
    if not ffmpeg_available(root):
        _install_ffmpeg(root, on_status)
    else:
        ensure_local_ffmpeg_on_path(root)
    return summarize_prerequisites(root)


def _ask_permission(title: str, message: str, parent=None) -> bool:
    try:
        from tkinter import messagebox
        import tkinter as tk
    except ImportError:
        return False
    created = False
    root = parent
    if root is None:
        root = tk.Tk()
        root.withdraw()
        created = True
    try:
        return bool(
            messagebox.askyesno(
                title,
                message,
                icon="question",
                parent=parent if parent is not None else root,
            )
        )
    finally:
        if created:
            root.destroy()


def request_prerequisite_install(
    root: str,
    parent=None,
    on_status: StatusFn | None = None,
    force: bool = False,
    install: bool = True,
) -> bool:
    """Show a consent dialog, then optionally download missing libraries.

    Returns False if declined. When ``install`` is False, a True return only
    means the user allowed the download (caller should run install on a worker).
    """
    status = summarize_prerequisites(root)
    if not force and status.startswith("All required"):
        if on_status:
            on_status(status)
        return True

    permitted = _ask_permission(
        "Install LiPAD libraries",
        "Project LiPAD needs Python packages and FFmpeg to run the desktop app, "
        "SSH to the Raspberry Pi, and receive the live MPEG-TS camera stream.\n\n"
        "Selecting Yes downloads:\n"
        "• customtkinter, pandas, pillow, paramiko\n"
        "• numpy, opencv-python, ultralytics, onnxruntime\n"
        "• FFmpeg (if it is not already installed)\n\n"
        "This uses the current Python interpreter and may take several minutes "
        "the first time (PyTorch is pulled in by ultralytics).",
        parent=parent,
    )
    if not permitted:
        if on_status:
            on_status("Library install permission was not granted.")
        return False
    if not install:
        return True
    if on_status:
        on_status("Installing required libraries…")
    try:
        result = install_prerequisites(root, on_status=on_status)
        if on_status:
            on_status(result)
        return True
    except Exception as exc:
        if on_status:
            on_status(f"Library install failed: {exc}")
        return False


def bootstrap_if_missing(root: str) -> None:
    """If the GUI cannot import, ask permission and install before continuing."""
    missing_gui = [
        pip_name
        for import_name, pip_name in PIP_PACKAGES
        if import_name in GUI_IMPORTS and importlib.util.find_spec(import_name) is None
    ]
    if not missing_gui:
        ensure_local_ffmpeg_on_path(root)
        return
    ok = request_prerequisite_install(root, parent=None, on_status=lambda msg: print(f"[SETUP] {msg}"))
    if not ok:
        print(
            "Required libraries are missing. Click Install libraries on the Home tab "
            "after installing customtkinter, pandas, and pillow, or run this app again and allow setup.",
            file=sys.stderr,
        )
        sys.exit(1)
    # New packages are not visible to this process until restart.
    os.execv(sys.executable, [sys.executable, *sys.argv])
