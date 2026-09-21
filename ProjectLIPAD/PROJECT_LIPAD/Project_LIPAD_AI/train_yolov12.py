from ultralytics import YOLO
import torch
import os

def train_lipad_from_scratch():
    print("[INFO] Checking hardware acceleration...")
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Training will execute on: {device.upper()}")

    # 1. Automatically detect exactly where this script is running from
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    yaml_config_file = os.path.join(base_dir, "dataset.yaml")
    yaml_blueprint = os.path.join(base_dir, "yolo12-seg.yaml")

    # Verify that your yolo12-seg.yaml blueprint is actually there
    if not os.path.exists(yaml_blueprint):
        print(f"\n[ERROR] Missing structural file at: {yaml_blueprint}")
        print("Please check that yolo12-seg.yaml is placed directly inside your Project_LIPAD_AI folder!")
        return

    print("[INFO] Dynamically generating dataset.yaml inside your workspace...")
    
    yaml_content = """path: datasets
train: images/train
val: images/val

names:
  0: crack
"""
    # Write the file directly inside your workspace
    with open(yaml_config_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[SUCCESS] dataset.yaml written cleanly at: {yaml_config_file}")

    print("[INFO] Initializing YOLOv12 Segmentation Architecture from blueprint...")
    model = YOLO(yaml_blueprint) 

    print("[INFO] Launching Training Session on CPU...")
    results = model.train(
        data=yaml_config_file,     # Points to the fresh workspace dataset.yaml
        epochs=15,                
        imgsz=320,                # Maintained at 320 for your Core i5 CPU processing limits
        batch=2,                  
        device=device,            
        workers=0,                # Required for Windows CPU multi-thread stability
        project="project_lipad",  
        name="yolov12_expert",    
        verbose=True
    )
    
    print("\n[SUCCESS] Project LIPAD YOLOv12 Expert Training Complete!")

if __name__ == "__main__":
    train_lipad_from_scratch()