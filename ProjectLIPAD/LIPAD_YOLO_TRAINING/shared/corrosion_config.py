"""Corrosion detection training and preprocessing configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Matches Roboflow export: 2 classes remapped, 3 dropped → 3 severity classes kept.
CORROSION_CLASS_NAMES: dict[int, str] = {
    0: "fair",
    1: "poor",
    2: "severe",
}

# Map raw label class ids (from an unprocessed export) to remapped ids.
# Drop any class id listed in CORROSION_DROP_CLASS_IDS.
CORROSION_CLASS_REMAP: dict[int, int] = {
    0: 0,  # fair
    1: 1,  # poor
    3: 2,  # severe (example: original id 3 remapped)
    4: 2,  # severe (example: original id 4 remapped)
}
CORROSION_DROP_CLASS_IDS: frozenset[int] = frozenset({2, 5, 6})


@dataclass(frozen=True)
class CorrosionPreprocessConfig:
    image_size: int = 640
    apply_auto_orient: bool = True
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    resize_mode: str = "stretch"  # Roboflow: Stretch to 640x640
    offline_augment_copies: int = 3  # Roboflow: 3 outputs per training example
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotation_degrees: float = 15.0
    exposure_fraction: float = 0.25  # ±25%
    blur_max_pixels: float = 2.5
    noise_max_pixel_fraction: float = 0.10  # up to 10% of pixels


@dataclass(frozen=True)
class CorrosionTrainConfig:
    epochs: int = 100
    lr0: float = 0.01
    optimizer: str = "SGD"
    imgsz: int = 640
    batch: int = 16
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    rotation_degrees: float = 15.0
    exposure_fraction: float = 0.25
    blur_max_pixels: float = 2.5
    noise_max_pixel_fraction: float = 0.10
    mosaic: float = 0.0  # disabled to align with Roboflow-style augmentations
    mixup: float = 0.0
    copy_paste: float = 0.0
    close_mosaic: int = 0


def corrosion_albumentations(config: CorrosionTrainConfig | CorrosionPreprocessConfig):
    """Custom Albumentations transforms for blur and noise (Ultralytics Python API)."""
    import albumentations as A

    blur_limit = 3  # Albumentations minimum kernel; closest to 2.5 px cap
    noise_std = config.noise_max_pixel_fraction

    return [
        A.Blur(blur_limit=(1, blur_limit), p=0.5),
        A.GaussNoise(std_range=(0.0, noise_std), p=0.5),
    ]


def corrosion_train_kwargs(
    config: CorrosionTrainConfig | None = None,
    *,
    batch: int | None = None,
    device: str | int | None = None,
    workers: int | None = None,
    project: str | None = None,
    name: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Build Ultralytics model.train() keyword arguments for corrosion."""
    cfg = config or CorrosionTrainConfig()
    kwargs: dict[str, Any] = {
        "epochs": cfg.epochs,
        "lr0": cfg.lr0,
        "optimizer": cfg.optimizer,
        "imgsz": cfg.imgsz,
        "batch": batch if batch is not None else cfg.batch,
        "fliplr": cfg.horizontal_flip_prob,
        "flipud": cfg.vertical_flip_prob,
        "degrees": cfg.rotation_degrees,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": cfg.exposure_fraction,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "mosaic": cfg.mosaic,
        "mixup": cfg.mixup,
        "copy_paste": cfg.copy_paste,
        "close_mosaic": cfg.close_mosaic,
        "augmentations": corrosion_albumentations(cfg),
        "resume": resume,
        "verbose": True,
    }
    if device is not None:
        kwargs["device"] = device
    if workers is not None:
        kwargs["workers"] = workers
    if project is not None:
        kwargs["project"] = project
    if name is not None:
        kwargs["name"] = name
    return kwargs


def config_summary() -> str:
    """Human-readable summary of active corrosion settings."""
    pre = CorrosionPreprocessConfig()
    train = CorrosionTrainConfig()
    lines = [
        "Base training",
        f"  epochs={train.epochs}, lr0={train.lr0}, optimizer={train.optimizer}",
        "Preprocessing",
        f"  auto_orient={pre.apply_auto_orient}",
        f"  resize={pre.image_size}x{pre.image_size} ({pre.resize_mode})",
        f"  clahe={pre.apply_clahe} (adaptive equalization)",
        f"  class_remap={len(CORROSION_CLASS_REMAP)} ids, drop={sorted(CORROSION_DROP_CLASS_IDS)}",
        f"  offline_augment_copies={pre.offline_augment_copies}",
        "Augmentations (training + optional offline preprocess)",
        f"  flips: horizontal + vertical",
        f"  rotation: ±{train.rotation_degrees}°",
        f"  exposure: ±{int(train.exposure_fraction * 100)}%",
        f"  blur: up to {train.blur_max_pixels}px",
        f"  noise: up to {int(train.noise_max_pixel_fraction * 100)}% of pixels",
        f"  classes: {CORROSION_CLASS_NAMES}",
    ]
    return "\n".join(lines)
