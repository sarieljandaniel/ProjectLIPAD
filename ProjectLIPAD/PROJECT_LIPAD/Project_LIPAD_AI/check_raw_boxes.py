import cv2
import os
from ultralytics import YOLO

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "yolov12_best.pt")
video_path = os.path.join(base_dir, "datasets", "OwnTestings")

# Find any available video
mp4_files = [f for f in os.listdir(video_path) if f.lower().endswith('.mp4')] if os.path.exists(video_path) else []
if not mp4_files:
    print("[ERROR] No video files found.")
    exit()

print(f"[INFO] Loading model: {model_path}")
model = YOLO(model_path)

user_file = input(f"\nType video name (Default: {mp4_files[0]}): ").strip()
target_video = os.path.join(video_path, user_file if user_file in mp4_files else mp4_files[0])

cap = cv2.VideoCapture(target_video)
print("\n[RUNNING] Displaying standard raw model outputs. Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    
    # Run absolute standard tracking with standard drawing functions
    results = model.track(frame, imgsz=320, verbose=False, persist=True, tracker="botsort.yaml")[0]
    
    # Force the model to use its native plotting layout (this ignores all our custom scripts)
    raw_render = results.plot()
    
    cv2.imshow("RAW YOLOv12 UNFILTERED OUTPUT", raw_render)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Stream closed.")