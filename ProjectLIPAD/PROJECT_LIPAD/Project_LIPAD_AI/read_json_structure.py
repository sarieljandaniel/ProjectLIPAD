import os
import json
from pathlib import Path

ann_dir = str(Path.home() / "Downloads" / "ds" / "ann")

if os.path.exists(ann_dir):
    json_files = [f for f in os.listdir(ann_dir) if f.lower().endswith('.json')]
    if json_files:
        sample_path = os.path.join(ann_dir, json_files[0])
        with open(sample_path, 'r') as f:
            data = json.load(f)
        
        print("--- SAMPLE JSON STRUCTURE KEYS ---")
        print(data.keys())
        print("\n--- FIRST OBJECT PREVIEW ---")
        # Let's print out what the objects look like so we can grab the coordinates safely
        if 'objects' in data:
            print(data['objects'][0] if len(data['objects']) > 0 else "Objects list is empty")
        elif 'annotation' in data:
            print(data['annotation'])
        else:
            print(str(data)[:500]) # Print first 500 characters as fallback
    else:
        print("[ERROR] No JSON files found in the directory.")
else:
    print(f"[ERROR] Directory not found: {ann_dir}")