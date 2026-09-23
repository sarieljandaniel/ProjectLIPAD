from pathlib import Path
from ultralytics import YOLO
import torch

def main():

    DETECT_DIR = Path(__file__).resolve().parent
    DATASET = str(DETECT_DIR / "dataset.yaml")
    MODEL = str(DETECT_DIR / "models" / "orig_50_best.pt")

    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    model = YOLO(MODEL)

    metrics = model.val(
        data=DATASET,
        split="val",
        imgsz=640,
        batch=16,
        workers=0,
        device=device,
        plots=True,
        save_json=True,
        project="evaluation_results",
        name="Original_50Epoch",
        exist_ok=True,
    )

    print(metrics)


if __name__ == "__main__":
    main()