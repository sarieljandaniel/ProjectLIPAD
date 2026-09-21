import cv2
import numpy as np
import os

# --- 1. SETUP & PATHS ---
# This tells the script exactly where your images are.
# Using './dataset' means "look in the dataset folder next to this script"
dataset_path = './dataset/Images' 

# Constants for the Deterministic Model
K_CONSTANT = 0.05  
N_EXPONENT = 0.5   
SIMULATED_DISTANCE = 500 # Assuming 500mm distance for the photos
FOCAL_LENGTH = 1400

# Get a list of all images in the folder
images = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# --- 2. THE MAIN LOOP ---
for img_name in images:
    path = os.path.join(dataset_path, img_name)
    frame = cv2.imread(path)
    
    # If the image fails to load, skip it
    if frame is None: 
        print(f"Failed to load {img_name}")
        continue

    # Resize massive internet images so they fit on your screen
    frame = cv2.resize(frame, (800, 600))

    # --- 3. ADAPTIVE LIGHTING (CLAHE) ---
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    frame_norm = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

    # --- 4. COLOR SPACE DETECTION (HSV) ---
    hsv = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2HSV)
    
    # THE RUST FILTER: Adjust these numbers if it misses rust or grabs wrong colors
    lower_rust = np.array([0, 118, 36])
    upper_rust = np.array([23, 255, 255])
    
    mask = cv2.inRange(hsv, lower_rust, upper_rust)

    # Clean up static/noise in the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- 5. MATH & DETERMINISTIC MODEL ---
    pixel_count = cv2.countNonZero(mask)
    actual_area = (pixel_count * (SIMULATED_DISTANCE**2)) / (FOCAL_LENGTH**2)
    forecast_30d = actual_area + (K_CONSTANT * (30**N_EXPONENT))

    # --- 6. DRAWING BOXES & TEXT ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 150: # Only box patches larger than 200 pixels
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Text Background (makes text easier to read)
    cv2.rectangle(frame, (5, 5), (400, 110), (0,0,0), -1)
    
    # Overlay Data
    cv2.putText(frame, f"File: {img_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Area: {actual_area:.2f} mm2", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"30D Forecast: {forecast_30d:.2f} mm2", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # --- 7. SHOW RESULT ---
    cv2.imshow("Original Mask (What the PC sees)", mask)
    cv2.imshow("Corrosion Detection System", frame)
    
    print(f"Showing {img_name}. Press 'N' to go to the next image, or 'Q' to quit.")
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()