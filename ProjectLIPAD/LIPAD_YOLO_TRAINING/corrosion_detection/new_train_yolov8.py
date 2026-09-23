"""Train YOLOv8 corrosion segmentation (Ameli et al. 2024 methodology)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.trainer import train

if __name__ == "__main__":
    train(
        task="corrosion",
        family="yolov8",

        # Paper settings
        epochs=250,
        imgsz=640,

        # Hardware limitation
        batch=2,

        # Optimization
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,
        warmup_epochs=5,
        warmup_momentum=0.8,
        weight_decay=0.0005,

        # Simulate paper batch behaviour
        nbs=8,

        # Reproducibility
        seed=0,
        deterministic=True,
        pretrained=True,

        # Device
        device=0,
        workers=2,

        # Augmentation
        mosaic=1.0,
        close_mosaic=10,
        fliplr=0.5,
        flipud=0.0,
        scale=0.5,
        translate=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,
        auto_augment="randaugment",

        resume=False,
        run_name="corrosion_yolov8m_seg_reproduction",
    )