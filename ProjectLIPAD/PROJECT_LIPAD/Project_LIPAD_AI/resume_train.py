from pathlib import Path

from ultralytics import YOLO

if __name__ == '__main__':
    last_weights = Path(__file__).resolve().parent / "runs" / "segment" / "train-2" / "weights" / "last.pt"
    model = YOLO(str(last_weights))
    
    # 2. Instruct the engine to resume from that exact checkpoint
    model.train(resume=True)