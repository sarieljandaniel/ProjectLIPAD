"""Train YOLOv12 crack segmentation. Run from LIPAD_YOLO_TRAINING root."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.trainer import train

if __name__ == "__main__":
    train("crack", "yolov12", epochs=100, imgsz=640, batch=8)
