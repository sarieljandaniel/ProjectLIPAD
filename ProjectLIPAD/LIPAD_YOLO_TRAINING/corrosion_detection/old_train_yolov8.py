"""Train YOLOv8 corrosion segmentation (Ameli et al. 2024 methodology).
Paper: Deep Learning-Based Steel Bridge Corrosion Segmentation and Condition 
Rating Using Mask RCNN and YOLOv8 (Infrastructures 2024, 9(1), 3).
https://doi.org/10.3390/infrastructures9010003 """

import sys 
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[1] 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT)) 

from shared.trainer import train 
if __name__ == "__main__": 
    train( 
        "corrosion", 
        "yolov8", 
        epochs=50, 
        imgsz=640, 
        batch=2, 
        lr0=0.0001, 
        device=0, 
        resume=False, 
        run_name="corrosion_yolov8m_seg",
        )