"""Offline preprocessing for corrosion datasets (Roboflow-style pipeline)."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps

from shared.corrosion_config import (
    CORROSION_CLASS_NAMES,
    CORROSION_CLASS_REMAP,
    CORROSION_DROP_CLASS_IDS,
    CorrosionPreprocessConfig,
)
from shared.paths import dataset_yaml_path, datasets_dir, task_root


def _load_yolo_seg_labels(label_path: Path) -> list[tuple[int, list[float]]]:
    rows: list[tuple[int, list[float]]] = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        cls_id = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
        rows.append((cls_id, coords))
    return rows


def _remap_labels(rows: list[tuple[int, list[float]]]) -> list[tuple[int, list[float]]]:
    remapped: list[tuple[int, list[float]]] = []
    for cls_id, coords in rows:
        if cls_id in CORROSION_DROP_CLASS_IDS:
            continue
        new_id = CORROSION_CLASS_REMAP.get(cls_id, cls_id)
        remapped.append((new_id, coords))
    return remapped


def _apply_clahe_bgr(image_bgr: np.ndarray, clip_limit: float, tile_grid: tuple[int, int]) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _maybe_augment(image_bgr: np.ndarray, cfg: CorrosionPreprocessConfig) -> np.ndarray:
    """Photometric-only offline aug so YOLO segment labels stay valid."""
    import albumentations as A

    blur_limit = 3  # Albumentations minimum kernel; closest to 2.5 px cap
    pipeline = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=cfg.exposure_fraction,
                contrast_limit=0.0,
                p=1.0,
            ),
            A.Blur(blur_limit=(1, blur_limit), p=0.7),
            A.GaussNoise(std_range=(0.0, cfg.noise_max_pixel_fraction), p=0.7),
        ]
    )
    return pipeline(image=image_bgr)["image"]


def _write_label(path: Path, rows: list[tuple[int, list[float]]]) -> None:
    lines = []
    for cls_id, coords in rows:
        coord_str = " ".join(f"{v:.6f}" for v in coords)
        lines.append(f"{cls_id} {coord_str}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_corrosion_dataset_yaml() -> Path:
    root = task_root("corrosion_detection")
    yaml_path = dataset_yaml_path("corrosion_detection")
    path_str = str(datasets_dir("corrosion_detection")).replace("\\", "/")
    content = {
        "path": path_str,
        "train": "images/train",
        "val": "images/val",
        "nc": len(CORROSION_CLASS_NAMES),
        "names": CORROSION_CLASS_NAMES,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def preprocess_corrosion_dataset(
    raw_dir: Path | str | None = None,
    *,
    config: CorrosionPreprocessConfig | None = None,
    clear_existing: bool = True,
) -> Path:
    """
    Prepare corrosion images/labels for YOLO training.

    Expected raw layout (when raw_dir is set):
      raw/images/train, raw/images/val, raw/labels/train, raw/labels/val

    When raw_dir is None, existing datasets/images/* is reprocessed in place.
    """
    cfg = config or CorrosionPreprocessConfig()
    base = datasets_dir("corrosion_detection")

    if raw_dir is not None:
        raw_root = Path(raw_dir)
        source_splits = {
            "train": (raw_root / "images" / "train", raw_root / "labels" / "train"),
            "val": (raw_root / "images" / "val", raw_root / "labels" / "val"),
        }
    else:
        source_splits = {
            "train": (base / "images" / "train", base / "labels" / "train"),
            "val": (base / "images" / "val", base / "labels" / "val"),
        }

    if clear_existing:
        for split in ("train", "val"):
            for sub in ("images", "labels"):
                target = base / sub / split
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    for split, (img_dir, lbl_dir) in source_splits.items():
        out_img_dir = base / "images" / split
        out_lbl_dir = base / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.is_dir():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in image_exts:
                continue

            pil = Image.open(img_path)
            if cfg.apply_auto_orient:
                pil = ImageOps.exif_transpose(pil)
            pil = pil.convert("RGB")
            orig_w, orig_h = pil.size

            pil = pil.resize((cfg.image_size, cfg.image_size), Image.Resampling.LANCZOS)
            image_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

            if cfg.apply_clahe:
                image_bgr = _apply_clahe_bgr(image_bgr, cfg.clahe_clip_limit, cfg.clahe_tile_grid_size)

            label_path = lbl_dir / f"{img_path.stem}.txt"
            rows = _remap_labels(_load_yolo_seg_labels(label_path))
            _ = orig_w, orig_h  # stretch resize keeps normalized polygon coords unchanged

            copies = 1 if split == "val" else max(1, cfg.offline_augment_copies)
            for copy_idx in range(copies):
                aug_image = image_bgr
                if copy_idx > 0:
                    aug_image = _maybe_augment(image_bgr, cfg)

                suffix = "" if copy_idx == 0 else f"_aug{copy_idx}"
                out_stem = f"{img_path.stem}{suffix}"
                out_img = out_img_dir / f"{out_stem}.jpg"
                out_lbl = out_lbl_dir / f"{out_stem}.txt"

                cv2.imwrite(str(out_img), aug_image)
                _write_label(out_lbl, rows)

    return write_corrosion_dataset_yaml()
