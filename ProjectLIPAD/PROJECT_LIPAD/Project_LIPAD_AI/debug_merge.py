import os
import json
from pathlib import Path

_NINJA_ROOT = Path.home() / "Downloads" / "ds"
NINJA_ANN_DIR = str(_NINJA_ROOT / "ann")
NINJA_IMG_DIR = str(_NINJA_ROOT / "img")

print("--- DIAGNOSTICS START ---")

# 1. Check if directories exist
print(f"ANN Dir Exists: {os.path.exists(NINJA_ANN_DIR)}")
print(f"IMG Dir Exists: {os.path.exists(NINJA_IMG_DIR)}")

if os.path.exists(NINJA_ANN_DIR):
    json_files = [f for f in os.listdir(NINJA_ANN_DIR) if f.lower().endswith('.json')]
    print(f"JSON Files Found: {len(json_files)}")
    
    if json_files:
        sample_json = json_files[0]
        base_name = os.path.splitext(sample_json)[0]
        print(f"Sample Base Name: {base_name}")
        
        # 2. Check what images are in the img folder to see formatting
        img_sample = os.listdir(NINJA_IMG_DIR)[:3] if os.path.exists(NINJA_IMG_DIR) else []
        print(f"Sample Files in IMG Folder: {img_sample}")
        
        # 3. Test a single file parse without try/except protection
        with open(os.path.join(NINJA_ANN_DIR, sample_json), 'r') as f:
            meta = json.load(f)
        
        print(f"JSON Structure Size Profile: {meta.get('size')}")
        objects = meta.get('objects', [])
        print(f"Objects found in sample: {len(objects)}")
        if objects:
            print(f"First object class: {objects[0].get('classTitle')}")
            print(f"First object has bitmap key: {'bitmap' in objects[0]}")

print("--- DIAGNOSTICS END ---")