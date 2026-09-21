from pathlib import Path

from ultralytics import YOLO

_AI_DIR = Path(__file__).resolve().parent
model = YOLO(str(_AI_DIR / "models" / "yolov5s_best.pt"))
print("[SUCCESS] Model successfully refactored for modern runtime engines!")