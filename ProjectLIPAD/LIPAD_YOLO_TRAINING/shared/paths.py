"""Path helpers — work locally (Windows) and on Google Colab."""

from __future__ import annotations

import os
from pathlib import Path


def is_colab() -> bool:
    try:
        import google.colab  # type: ignore

        return True
    except ImportError:
        return False


def training_repo() -> Path:
    """LIPAD_YOLO_TRAINING root (shared/paths.py -> parent)."""
    return Path(__file__).resolve().parents[1]


def _looks_like_placeholder(path: Path) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    return "full/path/to" in normalized


def _has_train_images(task_dir: Path) -> bool:
    train_dir = task_dir / "datasets" / "images" / "train"
    if not train_dir.is_dir():
        return False
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return any(p.suffix.lower() in image_exts for p in train_dir.iterdir() if p.is_file())


def task_root(task: str) -> Path:
    """Return crack_detection or corrosion_detection root."""
    repo = training_repo()
    default = repo / task

    env = os.environ.get("LIPAD_TASK_ROOT")
    if not env:
        return default

    candidate = Path(env).expanduser()
    if not candidate.is_absolute():
        candidate = (repo / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.exists() or _looks_like_placeholder(candidate):
        return default

    # Env points at the task folder (…/crack_detection).
    if _has_train_images(candidate):
        return candidate

    # Env points at LIPAD_YOLO_TRAINING repo root (Colab: /content/LIPAD_YOLO_TRAINING).
    task_dir = candidate / task
    if task_dir.is_dir() and _has_train_images(task_dir):
        return task_dir

    return default


def dataset_yaml_path(task: str) -> Path:
    return task_root(task) / "dataset.yaml"


def datasets_dir(task: str) -> Path:
    if is_colab() and task == "corrosion_detection":
        drive_dataset = Path(
            "/content/drive/MyDrive/"
            "LIPAD_TRAINING_VERSION2/datasets/corrosion/dataset"
        )

        if drive_dataset.exists():
            return drive_dataset

    return task_root(task) / "datasets"


def runs_dir(task: str) -> Path:
    return task_root(task) / "runs"


def ensure_dataset_layout(task: str) -> Path:
    base = datasets_dir(task)
    for sub in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base
