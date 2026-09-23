import os
# Keep offline mode active to avoid GitHub API errors
os.environ["YOLO_OFFLINE"] = "true"

from pathlib import Path
from ultralytics import YOLO

if __name__ == '__main__':
    last_weights = (
        Path(__file__).resolve().parent
        / "runs"
        / "yolov8"
        / "crack_yolov8_seg"
        / "weights"
        / "last.pt"
    )
    model = YOLO(str(last_weights))

    # 2. Kick off the resume and clamp it to your 50-epoch target
    # You don't need to re-state parameters (augment, close_mosaic, etc.) 
    # because YOLO reads them directly from the run's existing configuration logs.
    model.train(
        resume=True,
        epochs=50
    )