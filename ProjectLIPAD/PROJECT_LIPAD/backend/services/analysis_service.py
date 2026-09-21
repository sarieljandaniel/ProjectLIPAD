from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass, asdict
from pathlib import Path

from backend.config import (
    ANNOTATED_DIR,
    DATA_DIR,
    DEFAULT_WEIGHTS,
    ENGINE_SCRIPT,
    MORPH_CSV,
    UI_CSV,
)


@dataclass
class AnalysisState:
    status: str = "idle"
    message: str = "Ready"
    video_path: str | None = None
    inspection_type: str = "Crack"
    corrosion_env: str = "Wet"
    last_annotated_video: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class AnalysisService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._stop_requested = False
        self.state = AnalysisState()

    def get_state(self) -> dict:
        with self._lock:
            return self.state.to_dict()

    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None

    def stop(self) -> dict:
        with self._lock:
            proc = self._process
            if proc is None:
                self.state.status = "idle"
                self.state.message = "No analysis running."
                return self.state.to_dict()
            self._stop_requested = True
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        with self._lock:
            self._process = None
            self.state.status = "stopped"
            self.state.message = "Analysis stopped."
        return self.state.to_dict()

    def run(
        self,
        video_path: str,
        inspection_type: str = "Crack",
        corrosion_env: str = "Wet",
        gsd: float = 0.5436,
        frame_stride: int = 1,
        inference_width: int = 0,
    ) -> dict:
        video = Path(video_path)
        if not video.exists():
            self.state.status = "error"
            self.state.message = f"Video not found: {video_path}"
            return self.state.to_dict()

        with self._lock:
            if self._process is not None:
                self.state.message = "Engine already running."
                return self.state.to_dict()
            self._stop_requested = False

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
        base = video.stem
        annotated = ANNOTATED_DIR / f"{base}_annotated.mp4"

        cmd = [
            sys.executable,
            str(ENGINE_SCRIPT),
            "--video",
            str(video.resolve()),
            "--gsd",
            str(gsd),
            "--output_csv",
            str(MORPH_CSV),
            "--results_csv",
            str(UI_CSV),
            "--no_preview",
            "--frame_stride",
            str(max(1, frame_stride)),
            "--inference_width",
            str(max(0, inference_width)),
            "--output_video",
            str(annotated),
            "--inspection_type",
            inspection_type,
            "--corrosion_env",
            corrosion_env,
        ]
        if inspection_type.lower() == "crack":
            weights = DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.exists() else None
            if weights is None:
                self.state.status = "error"
                self.state.message = f"Weights missing: {DEFAULT_WEIGHTS}"
                return self.state.to_dict()
            cmd.extend(["--weights", str(weights)])

        self.state.status = "running"
        self.state.message = "Running analysis engine..."
        self.state.video_path = str(video.resolve())
        self.state.inspection_type = inspection_type
        self.state.corrosion_env = corrosion_env
        self.state.last_annotated_video = str(annotated)

        def _runner() -> None:
            try:
                with self._lock:
                    self._process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    proc = self._process
                code = proc.wait()
                with self._lock:
                    stopped = self._stop_requested
                    self._process = None
                    self._stop_requested = False
                if code != 0 and not stopped:
                    self.state.status = "error"
                    self.state.message = "Engine failed. Check logs and model weights."
                elif stopped:
                    self.state.status = "stopped"
                    self.state.message = "Analysis stopped."
                else:
                    self.state.status = "complete"
                    self.state.message = "Analysis complete. Open Analysis Overview."
            except Exception as exc:
                with self._lock:
                    self._process = None
                self.state.status = "error"
                self.state.message = f"Engine error: {exc}"

        threading.Thread(target=_runner, daemon=True).start()
        return self.state.to_dict()


analysis_service = AnalysisService()
