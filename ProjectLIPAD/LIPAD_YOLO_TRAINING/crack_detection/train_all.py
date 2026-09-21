"""Train all three YOLO families for crack detection sequentially."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.model_registry import all_families
from shared.trainer import train

if __name__ == "__main__":
    for family in all_families():
        print(f"\n{'=' * 60}\n  CRACK — {family.upper()}\n{'=' * 60}\n")
        train("crack", family, epochs=100, imgsz=640, batch=8)
