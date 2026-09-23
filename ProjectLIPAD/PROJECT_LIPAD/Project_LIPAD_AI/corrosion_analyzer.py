"""YOLOv8 corrosion segmentation inference and post-processing.

This module is used by ``lipad_runtime_engine_quantized.py`` whenever the
inspection target is Corrosion.  It deliberately keeps the existing function
signatures used by the desktop/live engine, while replacing the old HSV-only
pipeline with the trained YOLOv8m segmentation checkpoint.
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

K_CONSTANT = 0.05
N_EXPONENT = 0.5
MIN_CONTOUR_AREA_PX = 150
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "corrosion_models"
    / "yolov8m_baseline"
    / "weights"
    / "best.pt"
)

_model = None
_model_path: str | None = None


def _load_model(model_path: str | os.PathLike[str] | None = None):
    """Load the corrosion segmentation model once per engine process."""
    global _model, _model_path
    requested = str(model_path or MODEL_PATH)
    if _model is not None and _model_path == requested:
        return _model
    if not os.path.isfile(requested):
        raise FileNotFoundError(f"Corrosion model weights missing: {requested}")
    from ultralytics import YOLO

    _model = YOLO(requested, task="segment")
    _model_path = requested
    print(f"[SYSTEM] Corrosion YOLOv8m model loaded: {requested}")
    return _model


def _enhance_for_postprocess(frame: np.ndarray, environment: str) -> np.ndarray:
    """Prepare an image only for contour refinement; inference uses the raw frame."""
    if (environment or "Wet").strip().lower() == "dry":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def _postprocess_mask(mask: np.ndarray, frame: np.ndarray, environment: str) -> np.ndarray:
    """Clean a predicted mask and keep its boundary aligned to image edges."""
    binary = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Edge-guided refinement is constrained to the model mask, so it cannot
    # create detections outside the YOLO prediction.
    enhanced = _enhance_for_postprocess(frame, environment)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 75, 175)
    edge_band = cv2.dilate(edges, kernel, iterations=1)
    refined = cv2.bitwise_or(binary, cv2.bitwise_and(edge_band, binary))
    return cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)


def _severity_from_label(label: str, rar: float) -> str:
    name = (label or "").lower()
    if "severe" in name:
        return "Severe"
    if "poor" in name or "moderate" in name:
        return "Moderate"
    if "fair" in name or "minor" in name:
        return "Minor"
    if rar <= 0:
        return "None"
    if rar < 5:
        return "Minor"
    if rar < 15:
        return "Moderate"
    return "Severe"


def _detections(frame: np.ndarray, environment: str, model_path: str | None = None):
    model = _load_model(model_path)
    results = model.predict(source=frame, conf=0.25, iou=0.50, max_det=75, verbose=False)
    result = results[0]
    if result.masks is None or result.boxes is None:
        return []
    polygons = result.masks.xy
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.int().cpu().tolist()
    names = result.names or {}
    detections = []
    for idx, polygon in enumerate(polygons):
        if len(polygon) < 3:
            continue
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        mask = _postprocess_mask(mask, frame, environment)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA_PX:
            continue
        cls = classes[idx] if idx < len(classes) else 0
        label = str(names.get(cls, cls)) if isinstance(names, dict) else str(cls)
        detections.append((contour, boxes[idx], float(confs[idx]), label, mask))
    return detections


def _update_patch_stats(patch_stats, contour, label, confidence, next_patch_id):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None, next_patch_id
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    matched = None
    for pid, stats in patch_stats.items():
        if ((cx - stats["cx"]) ** 2 + (cy - stats["cy"]) ** 2) ** 0.5 < 80:
            matched = pid
            break
    if matched is None:
        matched = next_patch_id
        next_patch_id += 1
        patch_stats[matched] = {"areas_px": [], "cx": cx, "cy": cy, "frames": 0, "labels": [], "confidences": []}
    stats = patch_stats[matched]
    stats["areas_px"].append(float(cv2.contourArea(contour)))
    stats["cx"], stats["cy"] = cx, cy
    stats["frames"] += 1
    stats["labels"].append(label)
    stats["confidences"].append(float(confidence))
    return matched, next_patch_id


def annotate_corrosion_frame(frame, environment, patch_stats, next_patch_id):
    """Run YOLO segmentation, post-process masks, track patches, and annotate."""
    display = frame.copy()
    for contour, box, confidence, label, mask in _detections(frame, environment):
        pid, next_patch_id = _update_patch_stats(
            patch_stats, contour, label, confidence, next_patch_id
        )
        if pid is None:
            continue
        overlay = np.zeros_like(display)
        overlay[mask > 0] = (0, 140, 255)
        display = cv2.addWeighted(display, 1.0, overlay, 0.35, 0)
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.drawContours(display, [contour], -1, (0, 255, 255), 2)
        cv2.putText(display, f"Corrosion#{pid} {label} {confidence:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return display, next_patch_id


def patch_stats_to_rows(patch_stats, gsd_mm_per_px, environment, native_w, native_h, video_label, annotated=""):
    rows = []
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frame_area_mm2 = max(native_w * native_h * (gsd_mm_per_px ** 2), 1e-6)
    for pid in sorted(patch_stats):
        stats = patch_stats[pid]
        if not stats["areas_px"]:
            continue
        area_mm2 = float(np.max(stats["areas_px"])) * (gsd_mm_per_px ** 2)
        rar = area_mm2 / frame_area_mm2 * 100.0
        label = Counter(stats.get("labels", ["corrosion"])).most_common(1)[0][0]
        confidence = float(np.mean(stats.get("confidences", [0.0])))
        rows.append({
            "TimestampUTC": ts, "Video": video_label, "Annotated_Video": annotated,
            "Type": "Corrosion", "Environment": environment,
            "Severity": _severity_from_label(label, rar),
            "Crack_ID": f"Corrosion#{pid}", "Corrosion_Class": label,
            "Confidence": round(confidence, 4), "Avg_Length_mm": 0.0,
            "Avg_Width_mm": 0.0, "Avg_Orientation_Deg": 0.0,
            "Avg_Area_mm2": round(area_mm2, 2),
            "Forecast_30d_mm2": round(area_mm2 + K_CONSTANT * (30 ** N_EXPONENT), 2),
            "Rust_Area_Ratio_pct": round(rar, 2),
            "Total_Frames_Tracked": stats["frames"], "Critical_Shear_Alert": 0,
            "GSD_mm_per_px": float(gsd_mm_per_px),
        })
    return rows


def analyze_corrosion_video(video_path, gsd_mm_per_px, environment="Wet", frame_stride=5, max_frames=0, output_video=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = None
    if output_video:
        os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
        writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (native_w, native_h))
    patch_stats, next_id, frame_idx, processed = {}, 1, 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_stride > 1 and frame_idx % frame_stride != 0:
                continue
            processed += 1
            if max_frames and processed > max_frames:
                break
            display, next_id = annotate_corrosion_frame(frame, environment, patch_stats, next_id)
            if writer is not None:
                writer.write(display)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return patch_stats_to_rows(patch_stats, gsd_mm_per_px, environment, native_w, native_h,
                               os.path.abspath(video_path), os.path.abspath(output_video) if output_video else "")
