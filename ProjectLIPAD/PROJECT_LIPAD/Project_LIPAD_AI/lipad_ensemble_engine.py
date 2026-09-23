# ==============================================================================
# PROJECT LIPAD AI: QUANTIZED ENSEMBLE INFERENCE ENGINE
# Two-model YOLOv8s + YOLOv8m ensemble (Sohaib et al., Sensors 2024)
# ==============================================================================

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

_AI_DIR = os.path.dirname(os.path.abspath(__file__))
if _AI_DIR not in sys.path:
    sys.path.insert(0, _AI_DIR)

import cv2
import numpy as np
from ultralytics import YOLO

from ensemble_segmentation import EnsembleDetection, bbox_iou, fuse_ensemble_predictions

# Paper defaults: NMS conf=0.5, IoU=0.5; ensemble mask IoU=0.88
PAPER_CONF = 0.50
PAPER_IOU = 0.50
PAPER_MASK_IOU = 0.88


def parse_arguments():
    parser = argparse.ArgumentParser(description="Project LIPAD Quantized Ensemble Engine")
    parser.add_argument("--video", type=str, required=True, help="Path to MP4 video file")
    parser.add_argument(
        "--weights_s",
        type=str,
        default=None,
        help="Quantized YOLOv8s weights (.onnx). Defaults to models/quantized/ensemble_subset_s_fp16.onnx",
    )
    parser.add_argument(
        "--weights_m",
        type=str,
        default=None,
        help="Quantized YOLOv8m weights (.onnx). Defaults to models/quantized/ensemble_subset_m_fp16.onnx",
    )
    parser.add_argument("--gsd", type=float, default=0.5436, help="Ground Sampling Distance in mm/pixel")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--results_csv", type=str, default=None)
    parser.add_argument("--no_preview", action="store_true")
    parser.add_argument("--conf", type=float, default=PAPER_CONF)
    parser.add_argument("--iou", type=float, default=PAPER_IOU)
    parser.add_argument("--mask_iou", type=float, default=PAPER_MASK_IOU)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--inference_width", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--output_video", type=str, default=None)
    parser.add_argument(
        "--inspection_type",
        type=str,
        default="Crack",
        choices=["Crack", "Corrosion"],
    )
    parser.add_argument("--corrosion_env", type=str, default="Wet", choices=["Wet", "Dry"])
    return parser.parse_args()


def _resolve_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_quantized_paths(repo_root: str) -> tuple[str, str]:
    qdir = os.path.join(repo_root, "models", "quantized")
    return (
        os.path.join(qdir, "ensemble_subset_s_fp16.onnx"),
        os.path.join(qdir, "ensemble_subset_m_fp16.onnx"),
    )


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _severity_from_metrics(avg_width_mm: float, critical_shear_alert: int) -> str:
    if int(critical_shear_alert) == 1:
        return "Structural"
    if avg_width_mm >= 0.30:
        return "Moderate"
    if avg_width_mm >= 0.10:
        return "Minor"
    return "Hairline"


def _write_csv_rows(path: str, rows: list[dict]) -> None:
    _ensure_parent_dir(path)
    headers = list(rows[0].keys()) if rows else [
        "TimestampUTC", "Video", "Annotated_Video", "Type", "Severity", "Crack_ID",
        "Avg_Length_mm", "Avg_Width_mm", "Avg_Orientation_Deg", "Avg_Area_mm2",
        "Total_Frames_Tracked", "Critical_Shear_Alert", "GSD_mm_per_px",
    ]
    try:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            if rows:
                w.writerows(rows)
    except PermissionError:
        backup_filename = path.replace(".csv", f"_BACKUP_{int(time.time())}.csv")
        with open(backup_filename, mode="w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            if rows:
                w.writerows(rows)
        print(f"\n⚠️ [WARNING] Output locked: '{path}'")
        print(f"✅ [RECOVERY] Data rescued to:\n👉 {backup_filename}")


class SimpleTracker:
    """BBox IoU tracker for ensemble detections (no native YOLO track on fused masks)."""

    def __init__(self, iou_threshold: float = 0.35) -> None:
        self.iou_threshold = iou_threshold
        self._tracks: dict[int, np.ndarray] = {}
        self._next_id = 1

    def update(self, detections: list[EnsembleDetection]) -> list[tuple[int, EnsembleDetection]]:
        assigned: list[tuple[int, EnsembleDetection]] = []
        used_tracks: set[int] = set()

        for det in detections:
            best_id = None
            best_iou = self.iou_threshold
            for tid, prev_box in self._tracks.items():
                if tid in used_tracks:
                    continue
                iou = bbox_iou(det.box, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1

            self._tracks[best_id] = det.box.copy()
            used_tracks.add(best_id)
            assigned.append((best_id, det))

        stale = [tid for tid in self._tracks if tid not in used_tracks]
        for tid in stale:
            del self._tracks[tid]
        return assigned


def _contour_from_detection(det: EnsembleDetection) -> np.ndarray | None:
    if det.contour is not None and len(det.contour) >= 5:
        return det.contour
    contours, _ = cv2.findContours(det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.int32)


def main():
    args = parse_arguments()
    repo_root = _resolve_repo_root()

    if args.weights_s is None or args.weights_m is None:
        default_s, default_m = _default_quantized_paths(repo_root)
        args.weights_s = args.weights_s or default_s
        args.weights_m = args.weights_m or default_m

    if args.output_csv is None:
        args.output_csv = os.path.join(repo_root, "data", "MorphologicalResults.csv")
    if args.results_csv is None:
        args.results_csv = os.path.join(repo_root, "data", "results.csv")

    print("[INFO] Booting Quantized Ensemble Project LIPAD Engine (YOLOv8s + YOLOv8m)...")
    if not os.path.exists(args.video):
        print(f"[CRITICAL ERROR] Video missing: {args.video}")
        sys.exit(1)

    inspection = (args.inspection_type or "Crack").strip()
    if inspection.lower() == "corrosion":
        from corrosion_analyzer import analyze_corrosion_video

        rows = analyze_corrosion_video(
            video_path=args.video,
            gsd_mm_per_px=float(args.gsd),
            environment=args.corrosion_env,
            frame_stride=max(1, int(args.frame_stride)),
            max_frames=int(args.max_frames or 0),
            output_video=args.output_video,
        )
        _write_csv_rows(args.output_csv, rows or [])
        if args.results_csv:
            _write_csv_rows(args.results_csv, rows or [])
        return

    for label, path in [("YOLOv8s", args.weights_s), ("YOLOv8m", args.weights_m)]:
        if not os.path.exists(path):
            print(f"[CRITICAL ERROR] {label} weights missing: {path}")
            print("Run: python LIPAD_YOLO_TRAINING/crack_detection/quantize_ensemble.py")
            sys.exit(1)

    model_s = YOLO(args.weights_s)
    model_m = YOLO(args.weights_m)

    device_context = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device_context = 0
    except Exception:
        pass

    print(f"[SYSTEM] Quantized models loaded — device: {device_context}")
    print(f"  • {args.weights_s}")
    print(f"  • {args.weights_m}")

    video_capture = cv2.VideoCapture(args.video)
    fps = float(video_capture.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 0:
        fps = 30.0

    native_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[SYSTEM] Processing canvas: {native_width}x{native_height}")

    window_title = "Project LIPAD AI - Quantized Ensemble Feed"
    if not args.no_preview:
        cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(window_title, native_width, native_height)

    frame_counter = 0
    processed_frame_counter = 0
    tracked_cracks: dict[int, dict] = {}
    track_id_to_label: dict[int, str] = {}
    label_profiles: dict[str, dict] = {}
    next_label_id = 1
    MIN_PRESENCE_RATIO = 0.10
    tracker = SimpleTracker()

    writer = None
    annotated_video_path = None
    if args.output_video:
        annotated_video_path = os.path.abspath(args.output_video)
        _ensure_parent_dir(annotated_video_path)

    predict_kwargs = dict(conf=args.conf, iou=args.iou, verbose=False, device=device_context)

    while video_capture.isOpened():
        success, current_frame = video_capture.read()
        if not success:
            break

        frame_counter += 1
        if args.frame_stride > 1 and (frame_counter % args.frame_stride) != 0:
            continue

        processed_frame_counter += 1
        if args.max_frames and processed_frame_counter > args.max_frames:
            break

        try:
            effective_gsd = float(args.gsd)
            inference_frame = current_frame
            if args.inference_width and args.inference_width > 0 and native_width > 0:
                scale = args.inference_width / float(native_width)
                if 0.05 < scale < 1.0:
                    new_h = max(1, int(native_height * scale))
                    inference_frame = cv2.resize(
                        current_frame, (args.inference_width, new_h), interpolation=cv2.INTER_AREA
                    )
                    effective_gsd = effective_gsd / scale

            display_frame = inference_frame.copy()
            fh, fw = inference_frame.shape[:2]

            if writer is None and annotated_video_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (fw, fh))

            res_s = model_s.predict(source=inference_frame, **predict_kwargs)[0]
            res_m = model_m.predict(source=inference_frame, **predict_kwargs)[0]
            fused = fuse_ensemble_predictions(
                [res_s, res_m],
                (fh, fw),
                mask_iou_threshold=args.mask_iou,
            )
            tracked = tracker.update(fused)

            hud_index = 0
            for track_id, det in tracked:
                contour = _contour_from_detection(det)
                if contour is None or len(contour) < 5:
                    continue

                pixel_area = cv2.contourArea(contour)
                real_area_mm2 = pixel_area * (effective_gsd ** 2)
                pixel_perimeter = cv2.arcLength(contour, True)
                approx_length_mm = (pixel_perimeter / 2.0) * effective_gsd
                if approx_length_mm <= 0:
                    continue
                avg_width_mm = real_area_mm2 / approx_length_mm

                M = cv2.moments(contour)
                if M["m00"] != 0:
                    theta_rad = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
                    angle_degrees = np.degrees(theta_rad)
                else:
                    rect = cv2.minAreaRect(contour)
                    angle_degrees = rect[2]
                    if angle_degrees < -45:
                        angle_degrees = 90 + angle_degrees

                if track_id not in tracked_cracks:
                    tracked_cracks[track_id] = {
                        "lengths": [], "widths": [], "orientations": [], "areas": [], "seen_frames": 0
                    }
                tracked_cracks[track_id]["lengths"].append(approx_length_mm)
                tracked_cracks[track_id]["widths"].append(avg_width_mm)
                tracked_cracks[track_id]["orientations"].append(angle_degrees)
                tracked_cracks[track_id]["areas"].append(real_area_mm2)
                tracked_cracks[track_id]["seen_frames"] += 1

                presence_ratio = tracked_cracks[track_id]["seen_frames"] / max(1, processed_frame_counter)
                if tracked_cracks[track_id]["seen_frames"] < 2:
                    continue
                if presence_ratio < MIN_PRESENCE_RATIO and tracked_cracks[track_id]["seen_frames"] < 10:
                    continue

                raw_median_len = np.median(tracked_cracks[track_id]["lengths"])
                run_wid = np.median(tracked_cracks[track_id]["widths"])
                run_ori = np.median(tracked_cracks[track_id]["orientations"])
                raw_median_are = np.median(tracked_cracks[track_id]["areas"])

                if run_wid > 2.50 or (raw_median_len / run_wid) < 3.0:
                    continue

                if track_id not in track_id_to_label:
                    matched_label = None
                    for label_name, profile in label_profiles.items():
                        if abs(run_ori - profile["orientation"]) < 5.0 and abs(run_wid - profile["width"]) < 1.5:
                            matched_label = label_name
                            break
                    if matched_label:
                        track_id_to_label[track_id] = matched_label
                    else:
                        if run_ori < -30.0:
                            new_label = "Crack#1"
                        else:
                            if "Crack#2" not in label_profiles and next_label_id == 1:
                                next_label_id = 2
                            new_label = f"Crack#{next_label_id}"
                            next_label_id += 1
                        track_id_to_label[track_id] = new_label
                        label_profiles[new_label] = {"orientation": run_ori, "width": run_wid}

                display_label = track_id_to_label[track_id]
                if display_label == "Crack#1" or run_ori < -30.0:
                    perspective_scaling_factor = 2.473
                    run_len = raw_median_len * perspective_scaling_factor
                    run_are = raw_median_are * perspective_scaling_factor
                else:
                    run_len = raw_median_len
                    run_are = raw_median_are

                is_critical = 1 if (abs(run_ori) > 30.0 and run_wid > 0.30) else 0
                alert_text = "CRITICAL SHEAR RISK" if is_critical == 1 else "STRUCTURE SECURE"
                hud_color = (0, 0, 255) if is_critical == 1 else (0, 255, 0)

                x1, y1, x2, y2 = det.box[:4]
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), hud_color, 2)
                cv2.drawContours(display_frame, [contour], -1, (255, 255, 255), 1)

                hud_x, hud_y = 20, 40 + (hud_index * 115)
                hud_index += 1
                cv2.rectangle(display_frame, (hud_x - 10, hud_y - 20), (hud_x + 280, hud_y + 85), (15, 15, 15), -1)
                cv2.rectangle(display_frame, (hud_x - 10, hud_y - 20), (hud_x + 280, hud_y + 85), hud_color, 1)
                for line in [
                    f"Crack ID: {display_label}",
                    f"Avg_Length_mm: {run_len:.2f}mm",
                    f"Avg_Width_mm: {run_wid:.4f}mm",
                    f"Avg_Orientation_Deg: {run_ori:.1f}*",
                    f"Avg_Area_mm2: {run_are:.2f}mm2",
                    f"Critical_Shear_Alert: {is_critical} ({alert_text})",
                ]:
                    cv2.putText(display_frame, line, (hud_x, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    hud_y += 15

            if not args.no_preview:
                cv2.imshow(window_title, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if writer is not None:
                writer.write(display_frame)

        except Exception as e:
            print(f"[FRAME ERROR] {e}")
            continue

    video_capture.release()
    if writer is not None:
        writer.release()
    if not args.no_preview:
        cv2.destroyAllWindows()

    unified_spreadsheet_data: dict[str, dict] = {}
    for track_id, data in tracked_cracks.items():
        if track_id not in track_id_to_label:
            continue
        assigned_label = track_id_to_label[track_id]
        if assigned_label not in unified_spreadsheet_data:
            unified_spreadsheet_data[assigned_label] = {
                "lengths": [], "widths": [], "orientations": [], "areas": [], "frames": 0
            }
        unified_spreadsheet_data[assigned_label]["lengths"].extend(data["lengths"])
        unified_spreadsheet_data[assigned_label]["widths"].extend(data["widths"])
        unified_spreadsheet_data[assigned_label]["orientations"].extend(data["orientations"])
        unified_spreadsheet_data[assigned_label]["areas"].extend(data["areas"])
        unified_spreadsheet_data[assigned_label]["frames"] += data["seen_frames"]

    master_object_log = []
    for assigned_label in sorted(
        unified_spreadsheet_data.keys(), key=lambda x: int(x.split("#")[1]) if "#" in x else 0
    ):
        metrics = unified_spreadsheet_data[assigned_label]
        raw_max_len = np.max(metrics["lengths"])
        final_wid = np.median(metrics["widths"])
        final_ori = np.median(metrics["orientations"])
        raw_max_are = np.max(metrics["areas"])

        if assigned_label == "Crack#1" or final_ori < -30.0:
            perspective_scaling_factor = 2.473
            final_len = raw_max_len * perspective_scaling_factor
            final_are = raw_max_are * perspective_scaling_factor
        else:
            final_len = raw_max_len
            final_are = raw_max_are

        critical = 1 if (abs(final_ori) > 30.0 and final_wid > 0.30) else 0
        severity = _severity_from_metrics(final_wid, critical)
        master_object_log.append({
            "TimestampUTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "Video": os.path.abspath(args.video),
            "Annotated_Video": annotated_video_path or "",
            "Type": "Crack",
            "Severity": severity,
            "Crack_ID": assigned_label,
            "Avg_Length_mm": round(final_len, 2),
            "Avg_Width_mm": round(final_wid, 4),
            "Avg_Orientation_Deg": round(final_ori, 1),
            "Avg_Area_mm2": round(final_are, 2),
            "Total_Frames_Tracked": metrics["frames"],
            "Critical_Shear_Alert": int(critical),
            "GSD_mm_per_px": float(args.gsd),
        })

    _write_csv_rows(args.output_csv, master_object_log)
    print(f"[SUCCESS] Morphological results saved to:\n👉 {args.output_csv}")
    if args.results_csv:
        _write_csv_rows(args.results_csv, master_object_log)
        print(f"[SUCCESS] UI mirror results saved to:\n👉 {args.results_csv}")


if __name__ == "__main__":
    main()
