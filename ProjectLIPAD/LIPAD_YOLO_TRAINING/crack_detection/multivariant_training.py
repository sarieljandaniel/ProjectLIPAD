import os
from pathlib import Path
import yaml
from ultralytics import YOLO

def create_missing_yaml(subset_name, project_root):
    """
    Checks for the existence of a YOLO configuration file and creates it if missing.
    Ensures the training engine has the correct pointer to the disjoint datasets.
    """
    yaml_filename = f"{subset_name}.yaml"
    # Check if the file exists in the current directory
    if not os.path.exists(yaml_filename):
        print(f"[AUTO-GEN] Missing configuration detected. Creating: {yaml_filename}")
        
        yaml_content = {
            'path': 'datasets',
            'train': f'{subset_name}/images/train',
            'val': f'{subset_name}/images/val',
            'names': {0: 'crack'}
        }
        
        with open(yaml_filename, 'w') as yfile:
            yaml.dump(yaml_content, yfile, default_flow_style=False)
        print(f"[SUCCESS] {yaml_filename} generated next to the training script")

def train_ensemble_variants():
    # Define the project root for your datasets as per previous configuration [History]
    PROJECT_ROOT = str(Path(__file__).resolve().parent / "datasets")
    
    # Model configuration mapping (Model File : Subset Base Name)
    variants = {
        'yolov8s-seg.pt': 'subset_s', # Small variant
        'yolov8m-seg.pt': 'subset_m', # Medium variant
        'yolov8x-seg.pt': 'subset_x'  # Extra-Large variant
    }

    for model_weights, subset_base in variants.items():
        # Ensure the YAML file is present before attempting to load it
        create_missing_yaml(subset_base, PROJECT_ROOT)
        
        data_yaml = f"{subset_base}.yaml"
        print(f"\n[INIT] Starting training for {model_weights} using {data_yaml}...")
        
        # Load the specific YOLOv8 variant
        model = YOLO(model_weights)
        
        # Execute training with optimized hyperparameters for 89.62% precision [1-3]
        model.train(
            data=data_yaml,
            epochs=150,           # High Training Horizon [93, History]
            imgsz=640,            # High-Res for thin crack features [717, History]
            lr0=0.0001,           # Optimized Learning Rate for stable convergence [4, 5]
            optimizer='Adam',     # Robustness for complex concrete backgrounds [6]
            batch=2,              # VRAM Guardrail for NVIDIA GeForce MX230 [847, History]
            device=0,             # GPU acceleration enabled [7, 8]
            retina_masks=True,    # Pixel-perfect edge tracking for morphological math [History]
            close_mosaic=10,      # Disable final-stage mosaic for clean winding structures [9, 10]
            name=f"ensemble_{subset_base}"
        )

if __name__ == "__main__":
    train_ensemble_variants()