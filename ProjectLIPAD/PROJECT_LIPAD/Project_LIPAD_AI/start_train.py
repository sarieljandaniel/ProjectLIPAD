import os
from ultralytics import YOLO

if __name__ == '__main__':
    # Force the engine to locate the internal YOLOv12 segmentation architecture map
    # This initializes the model layers from scratch, ready for custom crack training!
    model = YOLO('yolov12n-seg.yaml') 
    
    # Kick off the training loop using your merged datasets
    model.train(
        data='dataset.yaml',
        epochs=110,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        close_mosaic=10
    )