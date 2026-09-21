"""Unified Ultralytics training entrypoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from shared.model_registry import ModelFamily, TaskType, all_families, get_model_spec
from shared.paths import dataset_yaml_path, ensure_dataset_layout, is_colab, runs_dir, task_root

try:
    from shared.corrosion_config import CorrosionTrainConfig, config_summary, corrosion_train_kwargs
    from shared.preprocess_corrosion import write_corrosion_dataset_yaml
except ImportError:
    CorrosionTrainConfig = None  # type: ignore[misc, assignment]
    config_summary = None  # type: ignore[assignment]
    corrosion_train_kwargs = None  # type: ignore[assignment]
    write_corrosion_dataset_yaml = None  # type: ignore[assignment]


def write_dataset_yaml(task: TaskType, class_name: str) -> Path:
    root = task_root(f"{task}_detection")
    ensure_dataset_layout(f"{task}_detection")
    yaml_path = dataset_yaml_path(f"{task}_detection")

    # Ultralytics prefers forward slashes
    path_str = str(root / "datasets").replace("\\", "/")
    content = {
        "path": path_str,
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def train(
    task: TaskType,
    family: ModelFamily,
    *,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    device: str | int | None = None,
    workers: int | None = None,
    project_name: str | None = None,
    run_name: str | None = None,
    resume: bool = False,
) -> Path:
    task_dir = f"{task}_detection"
    class_name = "crack" if task == "crack" else "corrosion"
    data_yaml = write_dataset_yaml(task, class_name)
    spec = get_model_spec(task, family)

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    if workers is None:
        workers = 2 if os.name == "nt" else 4
        if device == "cpu":
            workers = 0

    project = project_name or str(runs_dir(task_dir) / family)
    name = run_name or f"{class_name}_{family}_seg"

    print(f"[LiPAD] Task      : {task}")
    print(f"[LiPAD] Model     : {spec.weights} ({spec.description})")
    print(f"[LiPAD] Data      : {data_yaml}")
    print(f"[LiPAD] Device    : {device}")
    print(f"[LiPAD] Colab     : {is_colab()}")

    model = YOLO(spec.weights)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=project,
        name=name,
        resume=resume,
        close_mosaic=10,
        verbose=True,
    )

    best = Path(project) / name / "weights" / "best.pt"
    print(f"[LiPAD] Best weights: {best}")
    return best


def train_corrosion(
    family: ModelFamily,
    *,
    config: "CorrosionTrainConfig | None" = None,
    batch: int | None = None,
    device: str | int | None = None,
    workers: int | None = None,
    project_name: str | None = None,
    run_name: str | None = None,
    resume: bool = False,
) -> Path:
    """Train corrosion segmentation with Roboflow-aligned hyperparameters."""
    if corrosion_train_kwargs is None or write_corrosion_dataset_yaml is None:
        raise ImportError("shared.corrosion_config is required for train_corrosion()")

    task_dir = "corrosion_detection"
    data_yaml = write_corrosion_dataset_yaml()
    spec = get_model_spec("corrosion", family)

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    if workers is None:
        workers = 2 if os.name == "nt" else 4
        if device == "cpu":
            workers = 0

    project = project_name or str(runs_dir(task_dir) / family)
    name = run_name or f"corrosion_{family}_seg"

    print(config_summary())
    print(f"[LiPAD] Task      : corrosion")
    print(f"[LiPAD] Model     : {spec.weights} ({spec.description})")
    print(f"[LiPAD] Data      : {data_yaml}")
    print(f"[LiPAD] Device    : {device}")
    print(f"[LiPAD] Colab     : {is_colab()}")

    train_kwargs = corrosion_train_kwargs(
        config,
        batch=batch,
        device=device,
        workers=workers,
        project=project,
        name=name,
        resume=resume,
    )

    model = YOLO(spec.weights)
    model.train(data=str(data_yaml), **train_kwargs)

    best = Path(project) / name / "weights" / "best.pt"
    print(f"[LiPAD] Best weights: {best}")
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LiPAD YOLO segmentation models")
    p.add_argument("--task", choices=["crack", "corrosion"], required=True)
    p.add_argument("--model", choices=all_families(), required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="cuda device id, 'cpu', or omit for auto")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device: str | int | None
    if args.device is None:
        device = None
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = int(args.device)
    train(
        args.task,
        args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
