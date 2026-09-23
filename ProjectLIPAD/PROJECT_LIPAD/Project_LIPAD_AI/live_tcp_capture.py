"""Low-latency MPEG-TS TCP/UDP receiver for IMX519 / rpicam-vid.

TCP PC side equivalent:
  ffplay -listen 1 -i tcp://0.0.0.0:5000 -fflags nobuffer -flags low_delay

UDP PC side equivalent:
  ffplay -i udp://0.0.0.0:5000 -fflags nobuffer -flags low_delay -framedrop

Pi side sends to the selected protocol, for example:
  rpicam-vid ... --libav-format mpegts -o udp://<PC_IP>:5000

A dedicated drain thread always keeps only the newest decoded frame so inference
never sits on a growing queue.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Protocol

import cv2
import numpy as np


def _find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = (
        os.path.join(here, "..", "tools", "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(here, "..", "tools", "ffmpeg", "bin", "ffmpeg"),
        os.path.join(here, "..", "tools", "ffmpeg", "ffmpeg.exe"),
    )
    for candidate in candidates:
        path = os.path.abspath(candidate)
        if os.path.isfile(path):
            bin_dir = os.path.dirname(path)
            path_env = os.environ.get("PATH", "")
            if bin_dir.lower() not in path_env.lower().split(os.pathsep):
                os.environ["PATH"] = bin_dir + os.pathsep + path_env
            return path
    return None


def _normalize_protocol(protocol: str) -> str:
    normalized = (protocol or "tcp").strip().lower()
    if normalized not in {"tcp", "udp"}:
        raise ValueError("protocol must be either 'tcp' or 'udp'")
    return normalized


def build_rpicam_command(
    pc_ip: str,
    port: int,
    width: int = 1280,
    height: int = 720,
    bitrate: int = 3_000_000,
    protocol: str = "tcp",
) -> str:
    protocol = _normalize_protocol(protocol)
    return (
        "rpicam-vid -t 0 "
        f"--width {int(width)} --height {int(height)} "
        f"--bitrate {int(bitrate)} --inline "
        "--codec libav --libav-format mpegts "
        f"-o {protocol}://{pc_ip}:{int(port)}"
    )


class FrameSource(Protocol):
    eof: bool

    def read(self, timeout: float = 1.0) -> tuple[bool, np.ndarray | None]:
        ...

    def close(self) -> None:
        ...


class FileFrameSource:
    """Sequential file reader (no frame dropping)."""

    def __init__(self, path: str) -> None:
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self.eof = False
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0)
        self.fps = fps if fps > 0 else 30.0

    def read(self, timeout: float = 1.0) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._cap.read()
        if not ok:
            self.eof = True
            return False, None
        return True, frame

    def close(self) -> None:
        self._cap.release()


class LiveTcpFrameSource:
    """Receive a live MPEG-TS TCP or UDP stream and expose only the latest frame."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        width: int = 1280,
        height: int = 720,
        protocol: str = "tcp",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.width = int(width)
        self.height = int(height)
        self._protocol = _normalize_protocol(protocol)
        self.fps = 30.0
        self.eof = False
        self._frame_size = self.width * self.height * 3
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._running = True
        self._proc: subprocess.Popen | None = None
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._opencv_deferred = False
        self._ffmpeg_bin = _find_ffmpeg()
        self._use_ffmpeg = self._ffmpeg_bin is not None

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "fflags;nobuffer+discardcorrupt|flags;low_delay|"
            "probesize;32|analyzeduration;0|fifo_size;0|overrun_nonfatal;1"
        )

        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        os.makedirs(log_dir, exist_ok=True)
        self._ffmpeg_log = open(
            os.path.join(log_dir, "ffmpeg_live.log"),
            "w",
            encoding="utf-8",
            errors="replace",
        )
        try:
            if self._use_ffmpeg:
                try:
                    self._start_ffmpeg()
                except (OSError, RuntimeError) as exc:
                    print(
                        f"[LIVE] ffmpeg spawn failed ({exc}). "
                        "Opening OpenCV live capture in the background so the Pi can still connect.",
                        flush=True,
                    )
                    self._use_ffmpeg = False
                    self._opencv_deferred = True
            else:
                print("[LIVE] ffmpeg not on PATH — using OpenCV live capture.", flush=True)
                self._opencv_deferred = True
        except Exception:
            self._ffmpeg_log.close()
            raise

        self._thread = threading.Thread(target=self._drain, daemon=True, name="imx519-drain")
        self._thread.start()
        if self._opencv_deferred:
            time.sleep(0.35)

    def _ffmpeg_cmd(self) -> list[str]:
        url = f"{self._protocol}://{self.host}:{self.port}"
        binary = self._ffmpeg_bin or "ffmpeg"
        cmd = [
            binary,
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer+discardcorrupt",
            "-flags",
            "low_delay",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
        ]
        # TCP must listen for the Pi to connect; UDP binds directly to its port.
        if self._protocol == "tcp":
            cmd.extend(["-listen", "1"])
        cmd.extend([
            "-i", url,
            "-an",
            "-map",
            "0:v:0",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-fps_mode",
            "passthrough",
            "pipe:1",
        ])
        return cmd

    def _start_ffmpeg(self) -> None:
        kwargs: dict = {
            "stdout": subprocess.PIPE,
            "stderr": self._ffmpeg_log,
            "stdin": subprocess.DEVNULL,
            "bufsize": 0,
        }
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP
            no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            flags |= no_window
            kwargs["creationflags"] = flags
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0
            kwargs["startupinfo"] = startupinfo
        self._proc = subprocess.Popen(self._ffmpeg_cmd(), **kwargs)
        if self._proc.stdout is None:
            raise RuntimeError("ffmpeg stdout pipe was not created")
        time.sleep(0.35)
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"ffmpeg exited immediately with code {self._proc.returncode}. "
                "Check data/ffmpeg_live.log."
            )

    def _start_opencv(self) -> None:
        if self._protocol == "tcp":
            url = f"tcp://{self.host}:{self.port}?listen=1"
        else:
            url = f"udp://{self.host}:{self.port}"
        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _readexact(self, n: int) -> bytes | None:
        stdout = self._proc.stdout if self._proc is not None else None
        if stdout is None:
            return None
        buf = bytearray()
        while self._running and len(buf) < n:
            chunk = stdout.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf) if len(buf) == n else None

    def _drain(self) -> None:
        try:
            if self._opencv_deferred and self._proc is None:
                mode = "TCP client" if self._protocol == "tcp" else "UDP packets"
                print(f"[LIVE] Waiting for IMX519 {mode} (OpenCV)…", flush=True)
                self._start_opencv()
            if self._proc is not None:
                while self._running:
                    raw = self._readexact(self._frame_size)
                    if raw is None:
                        break
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (self.height, self.width, 3)
                    ).copy()
                    with self._lock:
                        self._latest = frame
            elif self._cap is not None:
                while self._running:
                    grabbed = self._cap.grab()
                    if not grabbed:
                        if not self._cap.isOpened():
                            break
                        time.sleep(0.002)
                        continue
                    ok, frame = self._cap.retrieve()
                    if not ok or frame is None:
                        continue
                    h, w = frame.shape[:2]
                    if w and h:
                        self.width, self.height = int(w), int(h)
                    with self._lock:
                        self._latest = frame
            else:
                return
        finally:
            self.eof = True

    def read(self, timeout: float = 1.0) -> tuple[bool, np.ndarray | None]:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                frame = self._latest
                self._latest = None
            if frame is not None:
                return True, frame
            if self.eof and not self._running:
                return False, None
            if self.eof:
                return False, None
            if time.monotonic() >= deadline:
                return False, None
            time.sleep(0.001)

    def close(self) -> None:
        self._running = False
        self.eof = True
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        log = getattr(self, "_ffmpeg_log", None)
        if log is not None:
            try:
                log.close()
            except Exception:
                pass
            self._ffmpeg_log = None


def mark_listen_ready(path: str | None) -> None:
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("listening\n")


def open_video_source(
    video_path: str | None,
    live: bool,
    listen_host: str = "0.0.0.0",
    listen_port: int = 5000,
    stream_width: int = 1280,
    stream_height: int = 720,
    protocol: str = "tcp",
) -> FrameSource:
    protocol = _normalize_protocol(protocol)
    if live:
        print(
            f"[LIVE] Receiving IMX519 MPEG-TS on {protocol}://{listen_host}:{listen_port} "
            f"({stream_width}x{stream_height}, low_delay)"
        )
        return LiveTcpFrameSource(
            host=listen_host,
            port=listen_port,
            width=stream_width,
            height=stream_height,
            protocol=protocol,
        )
    if not video_path:
        raise ValueError("Video path is required unless --live is set")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video missing: {video_path}")
    return FileFrameSource(video_path)


class PreviewJpegPublisher:
    """Encode a downscaled JPEG off the inference path; drop if a write is in flight."""

    def __init__(self, path: str, max_width: int = 960) -> None:
        self.path = path
        self.max_width = max_width
        self._lock = threading.Lock()
        self._pending: np.ndarray | None = None
        self._running = True
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._thread = threading.Thread(target=self._writer, daemon=True, name="preview-jpeg")
        self._thread.start()

    def publish(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending = frame.copy()

    def close(self) -> None:
        self._running = False

    def _writer(self) -> None:
        tmp = self.path + ".tmp"
        while self._running:
            with self._lock:
                frame = self._pending
                self._pending = None
            if frame is None:
                time.sleep(0.008)
                continue
            h, w = frame.shape[:2]
            if w > self.max_width > 0:
                scale = self.max_width / float(w)
                frame = cv2.resize(
                    frame,
                    (self.max_width, max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            try:
                cv2.imwrite(tmp, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                os.replace(tmp, self.path)
            except Exception:
                pass
