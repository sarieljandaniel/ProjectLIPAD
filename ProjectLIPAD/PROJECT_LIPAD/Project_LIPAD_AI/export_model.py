from pathlib import Path

from ultralytics import YOLO

model = YOLO(str(Path(__file__).resolve().parent / "runs" / "segment" / "train-2" / "weights" / "best.pt"))
model.export(format='onnx')

print("[SUCCESS] Model exported to ONNX format. Ready for deployment!")