import os
import json
import base64
import zlib
import numpy as np
import cv2
import shutil
import random
from pathlib import Path

_AI_DIR = Path(__file__).resolve().parent
_NINJA_ROOT = Path.home() / "Downloads" / "ds"
NINJA_ANN_DIR = str(_NINJA_ROOT / "ann")
NINJA_IMG_DIR = str(_NINJA_ROOT / "img")
TARGET_YOLO_DIR = str(_AI_DIR / "datasets")

def convert_ninja_dataset():
    if not os.path.exists(NINJA_ANN_DIR) or not os.path.exists(NINJA_IMG_DIR):
        print("[ERROR] Source folders missing.")
        return

    json_files = [f for f in os.listdir(NINJA_ANN_DIR) if f.lower().endswith('.json')]
    print(f"[INFO] Found {len(json_files)} total annotation files. Preparing train/val split...")

    random.seed(42) 
    random.shuffle(json_files)
    split_idx = int(len(json_files) * 0.8)
    
    splits = {
        "train": json_files[:split_idx],
        "val": json_files[split_idx:]
    }

    for stage, files in splits.items():
        yolo_img_dest = os.path.join(TARGET_YOLO_DIR, "images", stage)
        yolo_lbl_dest = os.path.join(TARGET_YOLO_DIR, "labels", stage)
        
        os.makedirs(yolo_img_dest, exist_ok=True)
        os.makedirs(yolo_lbl_dest, exist_ok=True)
        
        print(f"[PROCESSING] Merging {len(files)} assets into '{stage.upper()}' split...")

        success_count = 0
        for json_name in files:
            json_path = os.path.join(NINJA_ANN_DIR, json_name)
            
            # Match image names from '001.jpg.json' -> '001.jpg'
            matched_img_name = json_name.replace('.json', '').replace('.JSON', '')
            base_name = os.path.splitext(matched_img_name)[0]

            shutil_src_img = os.path.join(NINJA_IMG_DIR, matched_img_name)
            if not os.path.exists(shutil_src_img):
                continue
                
            with open(json_path, 'r') as f:
                meta_data = json.load(f)
                
            img_w = meta_data['size']['width']
            img_h = meta_data['size']['height']
            yolo_lines = []
            
            for obj in meta_data.get('objects', []):
                if obj.get('classTitle') != 'crack' or 'bitmap' not in obj:
                    continue
                    
                try:
                    bitmap_meta = obj['bitmap']
                    bitmap_data = bitmap_meta['data']
                    origin_x, origin_y = bitmap_meta['origin']
                    
                    # 1. Decode base64
                    compressed_bytes = base64.b64decode(bitmap_data)
                    
                    # 2. Decompress zlib to get the naked PNG bytes
                    png_bytes = zlib.decompress(compressed_bytes)
                    
                    # 3. Convert bytes directly to a 1D uint8 array for OpenCV
                    np_arr = np.frombuffer(png_bytes, dtype=np.uint8)
                    
                    # 4. Use IMREAD_UNCHANGED flag to preserve the alpha/grayscale channel of the mask slice
                    decoded_slice = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    
                    if decoded_slice is None:
                        continue
                        
                    # Handle if PNG decodes as RGBA or multi-channel; extract just the alpha or gray mask
                    if len(decoded_slice.shape) == 3:
                        if decoded_slice.shape[2] == 4: # RGBA
                            mask_layer = decoded_slice[:, :, 3]
                        else: # RGB
                            mask_layer = cv2.cvtColor(decoded_slice, cv2.COLOR_BGR2GRAY)
                    else:
                        mask_layer = decoded_slice
                        
                    # 5. Locate mask boundaries inside the crop slice
                    contours, _ = cv2.findContours(mask_layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if cv2.contourArea(contour) < 5:
                            continue
                            
                        normalized_coords = []
                        for pt in contour:
                            global_x = origin_x + pt[0][0]
                            global_y = origin_y + pt[0][1]
                            
                            x_norm = max(0.0, min(1.0, global_x / img_w))
                            y_norm = max(0.0, min(1.0, global_y / img_h))
                            normalized_coords.append(f"{x_norm:.6f} {y_norm:.6f}")
                            
                        if normalized_coords:
                            yolo_lines.append(f"0 {' '.join(normalized_coords)}")
                except Exception:
                    continue
            
            # Save and copy files if coordinates exist
            if yolo_lines:
                label_file_path = os.path.join(yolo_lbl_dest, base_name + ".txt")
                
                with open(label_file_path, 'a') as f_out:
                    if os.path.exists(label_file_path) and os.path.getsize(label_file_path) > 0:
                        f_out.write('\n')
                    f_out.write('\n'.join(yolo_lines))
                    
                shutil.copy(shutil_src_img, os.path.join(yolo_img_dest, matched_img_name))
                success_count += 1

        print(f" -> Successfully merged {success_count} files into {stage.upper()}!")

    print("\n[SUCCESS] Datasets are completely integrated and unified!")

if __name__ == "__main__":
    convert_ninja_dataset()