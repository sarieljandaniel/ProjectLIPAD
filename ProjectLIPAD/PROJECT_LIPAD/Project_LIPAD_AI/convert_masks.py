import cv2
import os
import numpy as np
from tqdm import tqdm

from pathlib import Path

_AI_DIR = Path(__file__).resolve().parent
RAW_MASK_DIR = str(Path.home() / "Desktop" / "RAW_MASKS_ALL")
TRAIN_IMG_DIR = str(_AI_DIR / "datasets" / "images" / "train")
VAL_IMG_DIR = str(_AI_DIR / "datasets" / "images" / "val")
TRAIN_LABEL_DIR = str(_AI_DIR / "datasets" / "labels" / "train")
VAL_LABEL_DIR = str(_AI_DIR / "datasets" / "labels" / "val")

def mask_to_yolo_segmentation_labels(img_dir, label_dir):
    # Double check if the image directory actually exists before running
    if not os.path.exists(img_dir):
        print(f"\n[ERROR] The directory does not exist: {img_dir}")
        print("Please verify your folder structure in File Explorer!")
        return

    os.makedirs(label_dir, exist_ok=True)
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    converted_count = 0

    for img_name in tqdm(images, desc=f"Converting masks for {os.path.basename(img_dir)}"):
        base_name = os.path.splitext(img_name)[0]
        
        # Look for a mask with the same name (.png or .jpg)
        mask_path = os.path.join(RAW_MASK_DIR, f"{base_name}.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(RAW_MASK_DIR, f"{base_name}.jpg")
            
        if not os.path.exists(mask_path):
            # If no mask matches, we create an empty label file (represents a background image)
            open(os.path.join(label_dir, f"{base_name}.txt"), 'w').close()
            continue

        # Load mask in grayscale
        mask = cv2.imread(mask_path, 0)
        h_img, w_img = mask.shape

        # Step 1: Apply Otsu's Thresholding to remove grayscale noise/shadows
        _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 2: Find contours (edges) of the cracks
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yolo_annotations = []
        for cnt in contours:
            # Skip very small noise artifacts (area less than 10 pixels)
            if cv2.contourArea(cnt) < 10:
                continue
                
            # Shape restructuring for coordinate pairs
            points = cnt.reshape(-1, 2)
            
            # Normalize coordinates relative to image scale (0.0 to 1.0)
            normalized_points = []
            for pt in points:
                x_norm = pt[0] / w_img
                y_norm = pt[1] / h_img
                normalized_points.append(f"{x_norm:.6f} {y_norm:.6f}")
            
            # Flatten to a single spaced string row
            polygon_string = " ".join(normalized_points)
            
            # Format: [class_id] [x1] [y1] [x2] [y2] ...
            yolo_annotations.append(f"0 {polygon_string}")

        # Save to output .txt file
        with open(os.path.join(label_dir, f"{base_name}.txt"), "w") as f:
            f.write("\n".join(yolo_annotations))
            converted_count += 1

    print(f"Successfully generated {converted_count} Instance Segmentation label files in {label_dir}")

if __name__ == "__main__":
    print("Starting Mask-to-YOLOv12 Segmentation conversion...")
    mask_to_yolo_segmentation_labels(TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
    mask_to_yolo_segmentation_labels(VAL_IMG_DIR, VAL_LABEL_DIR)
    print("All segmentation polygon labels generated successfully!")