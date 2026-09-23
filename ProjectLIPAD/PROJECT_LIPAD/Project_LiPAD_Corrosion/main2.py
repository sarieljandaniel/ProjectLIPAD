import cv2
import numpy as np
import os

# --- 1. SETUP & PATHS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, 'dataset/Images')

# Constants for the Deterministic Model
K_CONSTANT = 0.05  
N_EXPONENT = 0.5   
SIMULATED_DISTANCE = 500 
FOCAL_LENGTH = 1400

images = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_name in images:
    path = os.path.join(dataset_path, img_name)
    frame = cv2.imread(path)
    if frame is None: continue

    frame = cv2.resize(frame, (800, 600))
    # Create a copy for the final display
    output_img = frame.copy()

    # --- 2. PRE-PROCESSING (Adaptive Lighting) ---
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    frame_norm = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

    # --- 3. SEGMENTATION (HSV Masking) ---
    hsv = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2HSV)
    
    # Tighter values to fix the "Grey Metal" issue from your first image
    lower_rust = np.array([3, 85, 37])   # Saturation at 100 filters grey metal
    upper_rust = np.array([13, 255, 200]) 
    
    mask = cv2.inRange(hsv, lower_rust, upper_rust)

    # Clean up the noise (Morphological Opening/Closing)
    kernel = np.ones((1,1), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # --- 4. THE SEGMENTATION OVERLAY ---
    # Create a green "tint" image the same size as our frame
    green_overlay = np.zeros_like(frame)
    green_overlay[:] = (0, 255, 0) # Green color in BGR

    # Use the mask to only take the green where there is rust
    segmentation_effect = cv2.bitwise_and(green_overlay, green_overlay, mask=mask)

    # Blend the green tint with the original image (Transparency)
    # alpha is original, beta is the mask tint
    cv2.addWeighted(segmentation_effect, 0.5, output_img, 1.0, 0, output_img)

    # --- 5. MATH (Deterministic Model) ---
    pixel_count = cv2.countNonZero(mask)
    actual_area = (pixel_count * (SIMULATED_DISTANCE**2)) / (FOCAL_LENGTH**2)
    forecast_30d = actual_area + (K_CONSTANT * (30**N_EXPONENT))

    # --- 6. DATA OVERLAY ---
    cv2.rectangle(output_img, (5, 5), (420, 110), (0,0,0), -1)
    cv2.putText(output_img, f"File: {img_name}", (10, 30), 1, 1.2, (255,255,255), 2)
    cv2.putText(output_img, f"Corrosion Area: {actual_area:.2f} mm2", (10, 65), 1, 1.2, (0,255,0), 2)
    cv2.putText(output_img, f"30D Forecast: {forecast_30d:.2f} mm2", (10, 100), 1, 1.2, (0,255,255), 2)

    # --- 7. DISPLAY ---
    cv2.imshow("Binary Mask", mask)
    cv2.imshow("Pixel-wise Segmentation", output_img)
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'): break

cv2.destroyAllWindows()