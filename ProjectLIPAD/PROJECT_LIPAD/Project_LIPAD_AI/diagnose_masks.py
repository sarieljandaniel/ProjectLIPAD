import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "yolov12_best.pt")
video_dir = os.path.join(base_dir, "datasets", "OwnTestings")

# 1. Initialize YOLOv12
print(f"[INFO] Loading model: {model_path}")
model = YOLO(model_path)

# 2. Scan and list all available MP4 files
print("\n================ AVAILABLE TEST VIDEOS ================")
if not os.path.exists(video_dir):
    print(f"[ERROR] Directory missing: {video_dir}")
    exit()

mp4_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]

if not mp4_files:
    print(f"[WARNING] No .mp4 files found inside {video_dir}")
    print("Please place your test videos there or enter a full path below.")
else:
    for index, file_name in enumerate(mp4_files, 1):
        print(f" [{index}] {file_name}")
print("=======================================================")

# 3. Prompt user for the file name
user_file = input("\nType the name of the file you want to run (e.g., videotest1.mp4): ").strip()

# Resolve the absolute video path
if user_file in mp4_files:
    video_path = os.path.join(video_dir, user_file)
elif os.path.exists(user_file):
    video_path = user_file  # Allows pasting a full raw file path if it's stored elsewhere
else:
    # If they typed a shorthand without the extension
    if not user_file.lower().endswith('.mp4') and f"{user_file}.mp4" in mp4_files:
        video_path = os.path.join(video_dir, f"{user_file}.mp4")
    else:
        print(f"[ERROR] Target file '{user_file}' could not be resolved.")
        exit()

print(f"\n[INFO] Successfully targeted video: {video_path}")
print("[INFO] Scanning frame-by-frame for the first detection trigger...")

# 4. Process the selected video
cap = cv2.VideoCapture(video_path)
frame_count = 0
found_detection = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    
    frame_count += 1
    
    # Track across frames
    results = model.track(frame, imgsz=320, verbose=False, persist=True, tracker="botsort.yaml")[0]
    
    if results.masks is not None and len(results.boxes) > 0:
        print(f"\n[FOUND] Crack detection activated in '{os.path.basename(video_path)}' at Frame #{frame_count}!")
        found_detection = True
        
        # Pull mask parameters directly
        raw_mask = (results.masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            bx, by, bw, bh = cv2.boundingRect(largest_contour)
            aspect_ratio = float(bw) / bh if bh != 0 else 0
            
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area != 0 else 0
            
            print("\n================== DETECTED BLOB GEOMETRY REPORT ==================")
            print(f"Target Video File:     {os.path.basename(video_path)}")
            print(f"Trigger Frame Number:  {frame_count}")
            print(f"Total Mask Pixels:     {np.sum(raw_mask)}")
            print(f"Largest Contour Area:  {area}")
            print(f"Perimeter Length:      {perimeter:.2f} pixels")
            print(f"Blob Matrix Envelope:  Width = {bw} | Height = {bh}")
            print(f"Aspect Ratio (W/H):    {aspect_ratio:.2f}")
            print(f"Solidity Metric Value: {solidity:.4f}")
            print("===================================================================")
        else:
            print("[INFO] Mask data layer was created, but zero contours could be parsed.")
        break

cap.release()

if not found_detection:
    print(f"\n[INFO] Scanned through all {frame_count} frames of '{os.path.basename(video_path)}', but no crack detections occurred.")