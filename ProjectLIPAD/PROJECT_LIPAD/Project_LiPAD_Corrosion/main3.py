"""Dual-stage corrosion pipeline: color Scout (Stage 1) + YOLO Expert (Stage 2)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --- Paths & constants ---
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
MODEL_PATH = REPO_ROOT / "LIPAD_YOLO_TRAINING" / "corrosion_detection" / "models" / "new_250_best.pt"

K_CONSTANT = 0.05
N_EXPONENT = 0.5
SIMULATED_DISTANCE = 500  # mm
FOCAL_LENGTH = 1400

MORPH_KERNEL_SIZE = 5
MIN_SCOUT_AREA_PX = 150
MIN_VALID_AI_PIXELS = 50
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 640
DISPLAY_SIZE = (800, 600)


def resolve_dataset_path(base_dir: Path) -> Path:
    for name in ("Images", "images"):
        candidate = base_dir / "datasetCorrosion" / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No datasetCorrosion/Images folder under {base_dir}")


def get_severity(rar: float) -> tuple[str, tuple[int, int, int]]:
    if rar == 0:
        return "Rust-Free", (0, 255, 0)
    if rar < 5:
        return "Slight Rust", (0, 255, 255)
    if rar < 15:
        return "Medium Rust", (0, 165, 255)
    return "Severe Rust", (0, 0, 255)


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Y-channel equalization for contrast (Ref [60])."""
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def scout_color_mask(enhanced_img: np.ndarray) -> np.ndarray:
    """Stage 1: HSV + YCrCb(Otsu) intersection with 5x5 morph cleanup."""
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 100, 40])
    upper_hsv = np.array([25, 255, 200])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    ycrcb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2YCrCb)
    cr_channel = ycrcb[:, :, 1]
    _, mask_cr = cv2.threshold(cr_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    combined = cv2.bitwise_and(mask_hsv, mask_cr)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def scout_has_roi(scout_mask: np.ndarray, min_area: int = MIN_SCOUT_AREA_PX) -> bool:
    contours, _ = cv2.findContours(scout_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(cnt) >= min_area for cnt in contours)


def ai_expert_analysis(
    model: YOLO,
    frame: np.ndarray,
    scout_mask: np.ndarray,
    conf: float = DEFAULT_CONF,
    imgsz: int = DEFAULT_IMGSZ,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], int]:
    """Stage 2: segment only within Scout ROI; keep double-constrained pixels."""
    h, w = frame.shape[:2]
    validated_mask = np.zeros((h, w), dtype=np.uint8)
    validated_boxes: list[tuple[int, int, int, int]] = []

    roi_frame = cv2.bitwise_and(frame, frame, mask=scout_mask)
    results = model.predict(
        source=roi_frame,
        imgsz=imgsz,
        conf=conf,
        verbose=False,
        device=0 if torch.cuda.is_available() else "cpu",
    )

    for result in results:
        if result.masks is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        for idx, mask_tensor in enumerate(result.masks.data):
            mask = mask_tensor.cpu().numpy()
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            ai_mask = (mask > 0.5).astype(np.uint8) * 255

            constrained = cv2.bitwise_and(ai_mask, scout_mask)
            if cv2.countNonZero(constrained) < MIN_VALID_AI_PIXELS:
                continue

            validated_mask = cv2.bitwise_or(validated_mask, constrained)
            x1, y1, x2, y2 = boxes[idx].astype(int)
            validated_boxes.append((x1, y1, x2, y2))

    detections = len(validated_boxes)
    return validated_mask, validated_boxes, detections


def draw_hud(
    output_img: np.ndarray,
    img_name: str,
    severity_label: str,
    color_code: tuple[int, int, int],
    rar: float,
    actual_area_mm2: float,
    forecast_30d: float,
    detections: int,
) -> None:
    cv2.rectangle(output_img, (5, 5), (480, 185), (0, 0, 0), -1)
    cv2.putText(output_img, f"File: {img_name}", (15, 30), 1, 1.0, (255, 255, 255), 2)
    cv2.putText(output_img, "Mode: Scout + AI Expert", (15, 55), 1, 0.9, (200, 200, 200), 1)
    cv2.putText(output_img, f"Status: {severity_label}", (15, 80), 1, 1.2, color_code, 2)
    cv2.putText(output_img, f"RAR: {rar:.2f}%", (15, 110), 1, 1.0, (255, 255, 255), 1)
    cv2.putText(output_img, f"Validated patches: {detections}", (15, 135), 1, 1.0, (0, 255, 0), 1)
    cv2.putText(output_img, f"Current Area: {actual_area_mm2:.1f} mm2", (15, 160), 1, 1.0, (255, 255, 255), 1)
    cv2.putText(output_img, f"30D Forecast: {forecast_30d:.1f} mm2", (15, 185), 1, 1.0, (0, 255, 255), 2)


def process_image(
    model: YOLO,
    frame: np.ndarray,
    img_name: str,
    conf: float,
    show: bool = True,
) -> dict:
    frame = cv2.resize(frame, DISPLAY_SIZE)
    output_img = frame.copy()

    enhanced = enhance_frame(frame)
    scout_mask = scout_color_mask(enhanced)

    if scout_has_roi(scout_mask):
        final_mask, boxes, detections = ai_expert_analysis(model, frame, scout_mask, conf=conf)
    else:
        final_mask = np.zeros_like(scout_mask)
        boxes = []
        detections = 0

    total_pixels = frame.shape[0] * frame.shape[1]
    rust_pixels = cv2.countNonZero(final_mask)
    rar = (rust_pixels / total_pixels) * 100
    severity_label, color_code = get_severity(rar)

    actual_area_mm2 = (rust_pixels * (SIMULATED_DISTANCE**2)) / (FOCAL_LENGTH**2)
    forecast_30d = actual_area_mm2 + (K_CONSTANT * (30**N_EXPONENT))

    tint = np.zeros_like(frame)
    tint[:] = (0, 255, 0)
    segmentation = cv2.bitwise_and(tint, tint, mask=final_mask)
    cv2.addWeighted(segmentation, 0.55, output_img, 1.0, 0, output_img)

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    draw_hud(output_img, img_name, severity_label, color_code, rar, actual_area_mm2, forecast_30d, detections)

    if show:
        cv2.imshow("Stage 1: Scout Color Mask", scout_mask)
        cv2.imshow("Corrosion Segmentation System", output_img)

    return {
        "file": img_name,
        "severity": severity_label,
        "rar": rar,
        "detections": detections,
        "area_mm2": actual_area_mm2,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LiPAD dual-stage corrosion detection")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Image folder (default: datasetCorrosion/Images next to this script)",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="YOLO confidence threshold")
    parser.add_argument("--no-display", action="store_true", help="Process without GUI windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.source or resolve_dataset_path(BASE_DIR)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Loading AI Expert: {MODEL_PATH.name} (device={device})")
    model = YOLO(str(MODEL_PATH))

    images = sorted(
        f.name
        for f in dataset_path.iterdir()
        if f.suffix.lower() in {".jpg", ".png", ".jpeg", ".jfif", ".webp"}
    )
    if not images:
        raise FileNotFoundError(f"No images found in {dataset_path}")

    for img_name in images:
        path = dataset_path / img_name
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        stats = process_image(model, frame, img_name, conf=args.conf, show=not args.no_display)
        print(
            f"Processed {stats['file']}. "
            f"Severity: {stats['severity']} (RAR: {stats['rar']:.2f}%, "
            f"validated: {stats['detections']})"
        )

        if not args.no_display:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
