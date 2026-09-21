"""Project locations resolved from this file, not a machine-specific drive.

Clone or copy the repo anywhere: paths follow the folder layout next to this
module (PROJECT_LIPAD beside LIPAD_YOLO_TRAINING).
"""

from __future__ import annotations

from pathlib import Path

PROJECT_LIPAD_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_LIPAD_ROOT.parent
AI_DIR = PROJECT_LIPAD_ROOT / "Project_LIPAD_AI"
MODELS_DIR = PROJECT_LIPAD_ROOT / "models"
DEFAULT_ONNX_WEIGHTS = MODELS_DIR / "best.onnx"
DATA_DIR = PROJECT_LIPAD_ROOT / "data"
AI_DATASETS_DIR = AI_DIR / "datasets"
AI_DATASET_YAML = AI_DIR / "dataset.yaml"
YOLO_TRAINING_ROOT = WORKSPACE_ROOT / "LIPAD_YOLO_TRAINING"
CRACK_DETECTION_DIR = YOLO_TRAINING_ROOT / "crack_detection"
CRACK_DATASETS_DIR = CRACK_DETECTION_DIR / "datasets"
CORROSION_DETECTION_DIR = YOLO_TRAINING_ROOT / "corrosion_detection"
SUBSET_M_YAML = CRACK_DETECTION_DIR / "subset_m.yaml"
SUBSET_S_YAML = CRACK_DETECTION_DIR / "subset_s.yaml"
SUBSET_X_YAML = CRACK_DETECTION_DIR / "subset_x.yaml"


def as_str(path: Path) -> str:
    """Return a filesystem path string for the current OS."""
    return str(path)
