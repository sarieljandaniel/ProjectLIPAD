"""Train YOLOv11 corrosion segmentation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.trainer import train

if __name__ == "__main__":
    train("corrosion", "yolov11", epochs=100, imgsz=640, batch=8)
