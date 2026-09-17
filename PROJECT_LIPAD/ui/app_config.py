"""Centralized configuration for the desktop LiPAD application.

Keeping paths and defaults here prevents the UI class from being coupled to a
specific developer machine and gives future modules one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Repository-relative paths and runtime defaults used by the desktop app."""

    project_root: Path
    dist_ref_mm: float = 1168.4
    default_gsd: str = "0.5436"
    default_frame_stride: str = "1"
    default_inference_width: str = "0"
    default_live_port: str = "5000"
    default_live_protocol: str = "tcp"
    default_live_width: str = "1280"
    default_live_height: str = "720"
    default_live_bitrate: str = "3000000"
    default_pi_host: str = "lipad.local"
    default_pi_user: str = "lipad"

    @property
    def data_dir(self) -> Path:
        """Return the directory for generated previews, logs, and CSV results."""
        return self.project_root / "data"

    @property
    def engine_script(self) -> Path:
        """Return the quantized inference engine selected by the desktop app."""
        return self.project_root / "Project_LIPAD_AI" / "lipad_runtime_engine_quantized.py"

    @property
    def default_weights(self) -> Path:
        """Return the model location bundled with this checkout, when present."""
        return self.project_root / "models" / "best.onnx"

    @property
    def morph_results(self) -> Path:
        """Return the detailed morphological-results export path."""
        return self.data_dir / "MorphologicalResults.csv"

    @property
    def ui_results(self) -> Path:
        """Return the UI-friendly results export path."""
        return self.data_dir / "results.csv"

    @property
    def annotated_dir(self) -> Path:
        """Return the directory used for annotated videos."""
        return self.data_dir / "annotated"

    @classmethod
    def from_file(cls, file_path: str | Path) -> "AppConfig":
        """Build configuration from a desktop application module path."""
        return cls(project_root=Path(file_path).resolve().parent)
