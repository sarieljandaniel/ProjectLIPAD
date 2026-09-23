"""
Refactored Ensemble Segmentation Utility
Provides core IoU calculations and morphological operations for Post-NMS.
"""
import cv2
import numpy as np

def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Bounding Box IoU."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Mask IoU."""
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter / union) if union > 0 else 0.0

def morphological_refine(mask: np.ndarray) -> np.ndarray:
    """open-then-close: remove speckle, bridge small gaps along crack axis"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def edge_guided_snap(mask: np.ndarray, grayscale_image: np.ndarray, box: list) -> np.ndarray:
    """Phase 4: Edge-guided contour snap on the grayscale crop."""
    x1, y1, x2, y2 = map(int, box[:4])
    # Ensure coordinates are within frame bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(grayscale_image.shape[1], x2), min(grayscale_image.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return mask

    gray_crop = grayscale_image[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    # a. Run Canny on grayscale crop
    edges = cv2.Canny(gray_crop, 50, 150)
    
    # b. Intersect edge map with dilated YOLO mask
    dilated_mask = cv2.dilate(mask_crop, np.ones((5, 5), np.uint8), iterations=1)
    intersect = cv2.bitwise_and(edges, edges, mask=dilated_mask)
    
    # c. Dilate to rebuild a tighter mask and union with original mask crop
    snapped = cv2.dilate(intersect, np.ones((3, 3), np.uint8), iterations=1)
    refined_crop = cv2.bitwise_or(mask_crop, snapped)
    
    # Place back into the full-size mask
    snapped_mask = mask.copy()
    snapped_mask[y1:y2, x1:x2] = refined_crop
    return snapped_mask