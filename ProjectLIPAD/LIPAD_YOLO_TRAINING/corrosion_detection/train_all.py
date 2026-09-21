"""Train all three YOLO families for corrosion detection sequentially."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.model_registry import all_families
from shared.trainer import train

if __name__ == "__main__":
    for family in all_families():
        print(f"\n{'=' * 60}\n  CORROSION — {family.upper()}\n{'=' * 60}\n")
        train("corrosion", family, epochs=100, imgsz=640, batch=8)
