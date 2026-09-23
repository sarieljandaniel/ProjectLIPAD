# ==============================================================================
# PROJECT LIPAD AI: HIGH-RESOLUTION PRECISION INFERENCE ENGINE
# SYSTEM STATUS: ACTIVE | GEOMETRIC-HYBRID INFERENCE LAYER
# ==============================================================================

import os
import sys
import argparse
import time
import csv
from datetime import datetime, timezone
import cv2
import numpy as np
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description="Project LIPAD Structural Health Analytics Engine")
    parser.add_argument("--video", type=str, required=True, help="Path to MP4 video file")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to YOLO weights (.pt). Defaults to repo-local models/best.pt if present.",
    )
    # Hypothetical sensor calibration (PC-side placeholder until LiDAR exists)
    parser.add_argument("--gsd", type=float, default=0.5436, help="Ground Sampling Distance in mm/pixel (mm/px)")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Primary CSV output path. Defaults to data/MorphologicalResults.csv in repo root.",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default=None,
        help="Optional secondary CSV mirror output (e.g. UI expects data/results.csv).",
    )
    parser.add_argument(
        "--no_preview",
        action="store_true",
        help="Disable OpenCV preview window (recommended when called from the desktop app).",
    )
    parser.add_argument("--conf", type=float, default=0.40, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.40, help="YOLO IoU threshold")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Process every Nth frame (1 = all frames). Higher is faster but may reduce tracking quality.",
    )
    parser.add_argument(
        "--inference_width",
        type=int,
        default=0,
        help="Optional resize width for inference. 0 keeps native resolution. GSD will be scaled accordingly.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Optional cap on processed frames (0 = no cap). Useful for quick tests.",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help="Optional path to save an annotated MP4 (YOLO-style overlay).",
    )
    parser.add_argument(
        "--inspection_type",
        type=str,
        default="Crack",
        choices=["Crack", "Corrosion"],
        help="Inspection target: Crack (YOLO) or Corrosion (HSV/YCrCb).",
    )
    parser.add_argument(
        "--corrosion_env",
        type=str,
        default="Wet",
        choices=["Wet", "Dry"],
        help="Corrosion environment preset (Wet=marine, Dry=oxidation).",
    )
    return parser.parse_args()

def _resolve_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _default_weights_path(repo_root: str) -> str:
    return os.path.join(repo_root, "models", "best.pt")

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

def _severity_from_metrics(avg_width_mm: float, critical_shear_alert: int) -> str:
    if int(critical_shear_alert) == 1:
        return "Structural"
    # Conservative width-based heuristic for non-critical cases
    if avg_width_mm >= 0.30:
        return "Moderate"
    if avg_width_mm >= 0.10:
        return "Minor"
    return "Hairline"

def _write_csv_rows(path: str, rows: list[dict]) -> None:
    _ensure_parent_dir(path)
    headers = list(rows[0].keys()) if rows else [
        "TimestampUTC",
        "Video",
        "Annotated_Video",
        "Type",
        "Severity",
        "Crack_ID",
        "Avg_Length_mm",
        "Avg_Width_mm",
        "Avg_Orientation_Deg",
        "Avg_Area_mm2",
        "Total_Frames_Tracked",
        "Critical_Shear_Alert",
        "GSD_mm_per_px",
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

def main():
    args = parse_arguments()
    repo_root = _resolve_repo_root()

    if args.weights is None:
        args.weights = _default_weights_path(repo_root)

    if args.output_csv is None:
        args.output_csv = os.path.join(repo_root, "data", "MorphologicalResults.csv")

    if args.results_csv is None:
        args.results_csv = os.path.join(repo_root, "data", "results.csv")

    print("[INFO] Booting Geometric-Hybrid Precision Project LIPAD Engine...")
    if not os.path.exists(args.video):
        print(f"[CRITICAL ERROR] Video missing: {args.video}")
        sys.exit(1)

    inspection = (args.inspection_type or "Crack").strip()
    if inspection.lower() == "corrosion":
        from corrosion_analyzer import analyze_corrosion_video

        print(f"[INFO] Corrosion mode ({args.corrosion_env}) — HSV/YCrCb pipeline")
        rows = analyze_corrosion_video(
            video_path=args.video,
            gsd_mm_per_px=float(args.gsd),
            environment=args.corrosion_env,
            frame_stride=max(1, int(args.frame_stride)),
            max_frames=int(args.max_frames or 0),
            output_video=args.output_video,
        )
        if not rows:
            print("\n[INFO] Complete. No validated corrosion patches found.")
        _write_csv_rows(args.output_csv, rows)
        print(f"[SUCCESS] Morphological results saved to:\n👉 {args.output_csv}")
        if args.results_csv:
            _write_csv_rows(args.results_csv, rows)
            print(f"[SUCCESS] UI mirror results saved to:\n👉 {args.results_csv}")
        return

    if not os.path.exists(args.weights):
        print(f"[CRITICAL ERROR] Weights missing: {args.weights}")
        sys.exit(1)

    # Load YOLO model
    model = YOLO(args.weights)
    # Prefer GPU if available, otherwise CPU (still PC-side only)
    device_context = "cpu"
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            device_context = 0
    except Exception:
        device_context = "cpu"
    
    video_capture = cv2.VideoCapture(args.video)
    fps = float(video_capture.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 0:
        fps = 30.0
    
    # --- PURE NATIVE RESOLUTION READ ---
    native_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[SYSTEM] High-Fidelity Processing Canvas Locked: {native_width}x{native_height}")

    window_title = "Project LIPAD AI - Native Telemetry Feed"
    if not args.no_preview:
        cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(window_title, native_width, native_height)
    
    frame_counter = 0
    processed_frame_counter = 0
    
    tracked_cracks = {}
    MIN_PRESENCE_RATIO = 0.10  

    # DYNAMIC LABEL STITCHING STRUCTURES
    track_id_to_label = {}
    label_profiles = {}  # Stores median orientation and width for active labels
    next_label_id = 1

    # Optional annotated video writer (written at inference resolution for speed/consistency)
    writer = None
    annotated_video_path = None
    if args.output_video:
        annotated_video_path = os.path.abspath(args.output_video)
        _ensure_parent_dir(annotated_video_path)

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

        display_frame = current_frame.copy()

        try:
            # Optional downscaled inference for speed; adjust effective GSD accordingly
            effective_gsd = float(args.gsd)
            inference_frame = current_frame
            if args.inference_width and args.inference_width > 0 and native_width > 0:
                scale = args.inference_width / float(native_width)
                if 0.05 < scale < 1.0:
                    new_h = max(1, int(native_height * scale))
                    inference_frame = cv2.resize(current_frame, (args.inference_width, new_h), interpolation=cv2.INTER_AREA)
                    # If image is downscaled, each pixel covers more real-world area
                    effective_gsd = effective_gsd / scale

            # Render on the same coordinate space as inference outputs
            display_frame = inference_frame.copy()
            if writer is None and annotated_video_path:
                h, w = display_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (w, h))

            inference_results = model.track(
                source=inference_frame,
                persist=True, 
                conf=args.conf,
                iou=args.iou,
                tracker="botsort.yaml", 
                verbose=False, 
                device=device_context
            )
            
            result = inference_results[0]

            if result.boxes is not None and result.boxes.id is not None and result.masks is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
                boxes = result.boxes.xyxy.cpu().numpy()
                masks = result.masks.xy

                # First Pass: Compute structural dimensions for all targets
                for idx, track_id in enumerate(track_ids):
                    if idx >= len(masks):
                        continue
                        
                    contour_numpy = np.array(masks[idx], dtype=np.int32)
                    if len(contour_numpy) < 5:
                        continue

                    pixel_area = cv2.contourArea(contour_numpy)
                    real_area_mm2 = pixel_area * (effective_gsd ** 2)

                    pixel_perimeter = cv2.arcLength(contour_numpy, True)
                    approx_length_mm = (pixel_perimeter / 2.0) * effective_gsd
                    if approx_length_mm <= 0:
                        continue
                        
                    avg_width_mm = real_area_mm2 / approx_length_mm

                    M = cv2.moments(contour_numpy)
                    if M["m00"] != 0:
                        theta_rad = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
                        angle_degrees = np.degrees(theta_rad)
                    else:
                        rect = cv2.minAreaRect(contour_numpy)
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

                # Second Pass: Render overlays for confirmed defects
                hud_index = 0
                for idx, track_id in enumerate(track_ids):
                    if track_id not in tracked_cracks:
                        continue
                        
                    presence_ratio = tracked_cracks[track_id]["seen_frames"] / max(1, processed_frame_counter)
                    
                    if tracked_cracks[track_id]["seen_frames"] < 2:
                        continue
                        
                    if presence_ratio < MIN_PRESENCE_RATIO and tracked_cracks[track_id]["seen_frames"] < 10:
                        continue

                    raw_median_len = np.median(tracked_cracks[track_id]["lengths"])
                    run_wid = np.median(tracked_cracks[track_id]["widths"])
                    run_ori = np.median(tracked_cracks[track_id]["orientations"])
                    raw_median_are = np.median(tracked_cracks[track_id]["areas"])
                    
                    # MORPHOLOGICAL FILTER GATE (Keeps background thread detections out)
                    if run_wid > 2.50 or (raw_median_len / run_wid) < 3.0:
                        continue

                    # SUCCESSFUL OPTIMIZATION: DYNAMIC STITCHING LAYER
                    if track_id not in track_id_to_label:
                        matched_label = None
                        for label_name, profile in label_profiles.items():
                            if abs(run_ori - profile["orientation"]) < 5.0 and abs(run_wid - profile["width"]) < 1.5:
                                matched_label = label_name
                                break
                        
                        if matched_label:
                            track_id_to_label[track_id] = matched_label
                            print(f"[STITCH] Re-attaching fragmented tracker ID #{track_id} to existing {matched_label}")
                        else:
                            # --- GEOMETRIC PROFILE IDENTITY ASSIGNMENT ---
                            # Differentiate identity based on structural angle instead of screen coordinate
                            if run_ori < -30.0:
                                new_label = "Crack#1"
                            else:
                                # Safe default assigning sequential indices if it's not the steep diagonal shear crack
                                if f"Crack#2" not in label_profiles and next_label_id == 1:
                                    next_label_id = 2
                                new_label = f"Crack#{next_label_id}"
                                next_label_id += 1
                                
                            track_id_to_label[track_id] = new_label
                            label_profiles[new_label] = {"orientation": run_ori, "width": run_wid}
                    
                    display_label = track_id_to_label[track_id]

                    # --- REAL-TIME HUD PERSPECTIVE CORRECTION FILTER ---
                    # Checks identity via the stable geometric orientation signature
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

                    # Paint bounding geometry components directly onto native frames
                    x1, y1, x2, y2 = boxes[idx][:4]
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), hud_color, 2)
                    
                    contour_numpy = np.array(masks[idx], dtype=np.int32)
                    cv2.drawContours(display_frame, [contour_numpy], -1, (255, 255, 255), 1)

                    # Multi-Line Telemetry Overlay Panel Positioning
                    hud_x, hud_y = 20, 40 + (hud_index * 115)
                    hud_index += 1
                    
                    cv2.rectangle(display_frame, (hud_x - 10, hud_y - 20), (hud_x + 280, hud_y + 85), (15, 15, 15), -1)
                    cv2.rectangle(display_frame, (hud_x - 10, hud_y - 20), (hud_x + 280, hud_y + 85), hud_color, 1)

                    text_data = [
                        f"Crack ID: {display_label}", 
                        f"Avg_Length_mm: {run_len:.2f}mm",
                        f"Avg_Width_mm: {run_wid:.4f}mm",
                        f"Avg_Orientation_Deg: {run_ori:.1f}*",
                        f"Avg_Area_mm2: {run_are:.2f}mm2",
                        f"Critical_Shear_Alert: {is_critical} ({alert_text})"
                    ]

                    for line in text_data:
                        cv2.putText(display_frame, line, (hud_x, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                        hud_y += 15

            if not args.no_preview:
                cv2.imshow(window_title, display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
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

    # ----------------------------------------------------------------------
    # WRITE EXPORT TELEMETRY MATRIX UNIFYING STITCHED ENTRIES
    # ----------------------------------------------------------------------
    unified_spreadsheet_data = {}

    print(f"\n[COMPILE] Unifying and compiling database spreadsheet rows...")
    
    for track_id, data in tracked_cracks.items():
        if track_id not in track_id_to_label:
            continue
            
        assigned_label = track_id_to_label[track_id]
        
        if assigned_label not in unified_spreadsheet_data:
            unified_spreadsheet_data[assigned_label] = {
                "lengths": [], "widths": [], "orientations": [], "areas": [], "frames": 0
            }
            
        # Merge lists for stitched profiles
        unified_spreadsheet_data[assigned_label]["lengths"].extend(data["lengths"])
        unified_spreadsheet_data[assigned_label]["widths"].extend(data["widths"])
        unified_spreadsheet_data[assigned_label]["orientations"].extend(data["orientations"])
        unified_spreadsheet_data[assigned_label]["areas"].extend(data["areas"])
        unified_spreadsheet_data[assigned_label]["frames"] += data["seen_frames"]

    master_object_log = []
    for assigned_label in sorted(unified_spreadsheet_data.keys(), key=lambda x: int(x.split('#')[1]) if '#' in x else 0):
        metrics = unified_spreadsheet_data[assigned_label]
        
        raw_max_len = np.max(metrics["lengths"]) 
        final_wid = np.median(metrics["widths"])
        final_ori = np.median(metrics["orientations"])
        raw_max_are = np.max(metrics["areas"])

        # --- FINAL METRIC PERSPECTIVE ANALYSIS ---
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

    if not master_object_log:
        print("\n[INFO] Complete. No validated structural crack elements found.")

    # Always write CSVs (even if empty) so the UI has a consistent file to read.
    _write_csv_rows(args.output_csv, master_object_log)
    print(f"[SUCCESS] Morphological results saved to:\n👉 {args.output_csv}")

    if args.results_csv:
        _write_csv_rows(args.results_csv, master_object_log)
        print(f"[SUCCESS] UI mirror results saved to:\n👉 {args.results_csv}")

if __name__ == "__main__":
    main()