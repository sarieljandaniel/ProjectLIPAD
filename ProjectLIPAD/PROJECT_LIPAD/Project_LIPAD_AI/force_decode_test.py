import os
import json
import base64
import numpy as np
import cv2
from pathlib import Path

json_path = str(Path.home() / "Downloads" / "ds" / "ann" / "001.jpg.json")

print("--- FORCED DECODE TEST START ---")

with open(json_path, 'r') as f:
    meta_data = json.load(f)

img_w = meta_data['size']['width']
img_h = meta_data['size']['height']
print(f"Image Dimensions: {img_w}x{img_h}")

for idx, obj in enumerate(meta_data.get('objects', [])):
    print(f"\nObject [{idx}]: Class Title = '{obj.get('classTitle')}', Geometry = '{obj.get('geometryType')}'")
    
    if 'bitmap' in obj:
        try:
            bitmap_data = obj['bitmap']['data']
            origin = obj['bitmap']['origin']
            print(f" -> Origin: {origin}")
            print(f" -> Raw Base64 Data Length: {len(bitmap_data)} characters")
            
            # Let's see if base64 decoding actually works or fails
            raw_bytes = base64.b64decode(bitmap_data)
            print(f" -> Successfully decoded base64 string to {len(raw_bytes)} bytes")
            
            # Let's see if openCV can read this raw text chunk as an image layer
            np_data = np.frombuffer(raw_bytes, dtype=np.uint8)
            decoded_slice = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
            
            if decoded_slice is None:
                print(" -> [FAIL] cv2.imdecode returned None! The string isn't a standard image byte array.")
            else:
                print(f" -> [SUCCESS] Slice decoded! Shape: {decoded_slice.shape}")
                contours, _ = cv2.findContours(decoded_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(f" -> Found {len(contours)} contour paths inside this slice.")
                
        except Exception as e:
            print(f" -> [CRASH] Failed during parsing: {str(e)}")

print("\n--- FORCED DECODE TEST END ---")