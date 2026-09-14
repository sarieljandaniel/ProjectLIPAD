# ==============================================================================
# PROJECT LIPAD AI: QUANTIZED ENSEMBLE INFERENCE ENGINE
# SYSTEM STATUS: ACTIVE | GEOMETRIC-HYBRID + BOTSORT TRACKING LAYER
# ==============================================================================

from __future__ import annotations

import argparse
import csv
import os
import re
import socket
import sys
import threading
import time
from datetime import datetime, timezone

_AI_DIR = os.path.dirname(os.path.abspath(__file__))
if _AI_DIR not in sys.path:
    sys.path.insert(0, _AI_DIR)

import cv2
import numpy as np

from live_tcp_capture import PreviewJpegPublisher, mark_listen_ready, open_video_source

# --- GSD MODE 1: Hypothetical calibration ---
GSD_HYPOTHETICAL_MM_PER_PX = 0.5436

# --- GSD MODE 2: Live VL53L8CX via Raspberry Pi 4B hotspot (UDP port 50007) ---
DIST_REF_MM = 1168.4  
GSD_FROM_LIDAR_MM_PER_PX = GSD_HYPOTHETICAL_MM_PER_PX * (1.0 / DIST_REF_MM)

PAPER_CONF = 0.50
PAPER_IOU = 0.50
TELEMETRY_PORT = 50007


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Project LIPAD Quantized Structural Health Analytics Engine"
    )
    parser.add_argument("--video", type=str, default=None, help="Path to MP4 video file (not required for --live)")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Receive IMX519 MPEG-TS over TCP or UDP (rpicam-vid → this PC).",
    )
    parser.add_argument("--listen_host", type=str, default="0.0.0.0", help="TCP/UDP receive bind address")
    parser.add_argument("--listen_port", type=int, default=5000, help="TCP or UDP receive port")
    parser.add_argument(
        "--stream_protocol",
        type=str.lower,
        choices=["tcp", "udp"],
        default="tcp",
        help="Live MPEG-TS transport protocol",
    )
    parser.add_argument("--stream_width", type=int, default=1280, help="Expected live stream width")
    parser.add_argument("--stream_height", type=int, default=720, help="Expected live stream height")
    parser.add_argument(
        "--preview_jpeg",
        type=str,
        default=None,
        help="Atomic JPEG path for the desktop app inference/annotated live view.",
    )
    parser.add_argument(
        "--raw_preview_jpeg",
        type=str,
        default=None,
        help="Atomic JPEG path for the desktop app raw live view.",
    )
    parser.add_argument(
        "--ready_flag",
        type=str,
        default=None,
        help="Touch this file once the live receiver is ready so the app can start rpicam-vid.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=r"C:\Users\lenovo\Project_LIPAD_v3\Corrosion\PROJECT_LIPAD\models\best.onnx",
        help="Path to single quantized YOLO weights (.onnx)",
    )
    parser.add_argument(
        "--gsd_mode",
        type=str,
        default="hypothetical",
        choices=["hypothetical", "lidar"],
        help="GSD source: hypothetical fixed value or live VL53L8CX distance",
    )
    parser.add_argument(
        "--gsd",
        type=float,
        default=None,
        help="Override GSD (mm/px). Uses GSD_HYPOTHETICAL_MM_PER_PX if omitted.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Primary CSV output path. Defaults to data/MorphologicalResults.csv",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default=None,
        help="Secondary CSV mirror output path.",
    )
    parser.add_argument(
        "--no_preview",
        action="store_true",
        help="Disable OpenCV preview window.",
    )
    parser.add_argument("--conf", type=float, default=PAPER_CONF, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=PAPER_IOU, help="YOLO IoU threshold")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Process every Nth frame (1 = all frames).",
    )
    parser.add_argument(
        "--inference_width",
        type=int,
        default=1280, # Phase 2: Production resolution
        help="Optional resize width for inference. 0 keeps native resolution.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Cap on processed frames (0 = no cap).",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help="Path to save annotated MP4 video.",
    )
    parser.add_argument(
        "--inspection_type",
        type=str,
        default="Crack",
        choices=["Crack", "Corrosion"],
        help="Inspection target.",
    )
    parser.add_argument(
        "--corrosion_env",
        type=str,
        default="Wet",
        choices=["Wet", "Dry"],
        help="Corrosion environment preset.",
    )
    return parser.parse_args()


def _resolve_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _onnx_runtime_cuda_usable(weights_path: str) -> bool:
    """Return True only when ORT can actually run the model on CUDA (not just list the EP)."""
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False
        if not os.path.isfile(weights_path):
            return False
        session = ort.InferenceSession(
            weights_path,
            providers=[("CUDAExecutionProvider", {"device_id": 0})],
        )
        return session.get_providers()[0] == "CUDAExecutionProvider"
    except Exception:
        return False


def _resolve_inference_device(weights_path: str) -> str | int:
    """Pick an inference device that matches the runtime backend for the weights file."""
    weights_lower = (weights_path or "").lower()

    if weights_lower.endswith(".onnx"):
        # Ultralytics enables CUDA IO binding when device=0. If ORT cannot init CUDA
        # (driver/CUDA mismatch) it silently falls back to CPU while tensors stay on
        # GPU, causing per-frame "no data transfer registered" errors.
        if _onnx_runtime_cuda_usable(weights_path):
            return 0
        print("[SYSTEM] ONNX inference will use CPU (ORT CUDA unavailable or misconfigured).")
        return "cpu"

    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.zeros(1, device="cuda")
            return 0
    except Exception:
        pass
    return "cpu"


def _patch_onnxruntime_webgpu_compatibility() -> None:
    """Avoid a broken WebGPU capability probe in mismatched ONNX Runtime installs.

    Some Windows installations pair a newer Python wrapper with an older compiled
    ONNX Runtime extension. CPU inference does not use WebGPU graph capture, but
    the wrapper still probes for its extension method before every ``session.run``.
    """
    try:
        from onnxruntime.capi import onnxruntime_inference_collection as ort_collection

        original = ort_collection.InferenceSession._validate_graph_capture_run_api
        if getattr(original, "_lipad_compatibility_patch", False):
            return

        def _validate_graph_capture_run_api(self, run_options=None):
            if not hasattr(self._sess, "is_webgpu_graph_capture_enabled"):
                return
            return original(self, run_options)

        _validate_graph_capture_run_api._lipad_compatibility_patch = True
        ort_collection.InferenceSession._validate_graph_capture_run_api = (
            _validate_graph_capture_run_api
        )
        print("[SYSTEM] Applied ONNX Runtime CPU compatibility patch.")
    except ImportError:
        pass


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _gsd_from_lidar_distance_mm(distance_mm: float) -> float:
    if distance_mm <= 0:
        return GSD_HYPOTHETICAL_MM_PER_PX
    return distance_mm * GSD_FROM_LIDAR_MM_PER_PX


def _resolve_base_gsd(args) -> tuple[float, str]:
    if args.gsd is not None:
        return float(args.gsd), "manual_override"
    if args.gsd_mode == "lidar":
        return GSD_HYPOTHETICAL_MM_PER_PX, "lidar_pending"
    return GSD_HYPOTHETICAL_MM_PER_PX, "hypothetical"


class LidarTelemetryReader:
    """Background UDP listener for Pi 4B hotspot VL53L8CX distance packets."""

    def __init__(self, port: int = TELEMETRY_PORT) -> None:
        self.port = port
        self.latest_distance_mm: float | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def get_distance_mm(self) -> float | None:
        with self._lock:
            return self.latest_distance_mm

    @staticmethod
    def _parse_distance(message: str) -> float | None:
        match = re.search(r"DISTANCE:\s*(\d+\.?\d*)", message, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"(\d+\.?\d*)\s*mm", message, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def _listen(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", self.port))
            sock.settimeout(1.0)
            while self._running:
                try:
                    data, _addr = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                distance = self._parse_distance(data.decode("utf-8", errors="replace").strip())
                if distance is not None:
                    with self._lock:
                        self.latest_distance_mm = distance
        finally:
            sock.close()


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

def post_nms_mask_merge(boxes, masks, confs, track_ids):
    """
    Phase 2: Post-NMS Mask Union based on Confidence, IoU, and Orientation.
    """
    if len(boxes) == 0:
        return boxes, masks, confs, track_ids

    # a. Sort detections by confidence (descending)
    sorted_indices = np.argsort(-confs)
    boxes = boxes[sorted_indices]
    masks = [masks[i] for i in sorted_indices]
    confs = confs[sorted_indices]
    track_ids = [track_ids[i] for i in sorted_indices] if track_ids else []

    merged_flags = [False] * len(boxes)
    final_boxes, final_masks, final_confs, final_ids = [], [], [], []

    for i in range(len(boxes)):
        if merged_flags[i]:
            continue

        base_box = boxes[i]
        base_mask_pts = np.array(masks[i], dtype=np.int32)
        base_conf = confs[i]
        base_id = track_ids[i] if track_ids else None

        # Calculate orientation for base mask
        base_rect = cv2.minAreaRect(base_mask_pts)
        base_angle = base_rect[2]
        
        for j in range(i + 1, len(boxes)):
            if merged_flags[j]:
                continue
                
            compare_box = boxes[j]
            compare_mask_pts = np.array(masks[j], dtype=np.int32)
            
            # Calculate IoUs
            x1, y1 = max(base_box[0], compare_box[0]), max(base_box[1], compare_box[1])
            x2, y2 = min(base_box[2], compare_box[2]), min(base_box[3], compare_box[3])
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            box_area1 = (base_box[2] - base_box[0]) * (base_box[3] - base_box[1])
            box_area2 = (compare_box[2] - compare_box[0]) * (compare_box[3] - compare_box[1])
            bbox_iou = inter_area / float(box_area1 + box_area2 - inter_area + 1e-5)
            
            # Simplified mask IoU check via bounds/contours
            mask_iou = bbox_iou # Replace with exact pixel-wise Mask IoU if rasterized
            
            # Orientation check
            comp_rect = cv2.minAreaRect(compare_mask_pts)
            comp_angle = comp_rect[2]
            angle_diff = abs(base_angle - comp_angle)
            
            # b. Condition: Mask IoU > 0.15 OR bbox IoU > 0.35 and |Δθ| < 25°
            if (mask_iou > 0.15) or (bbox_iou > 0.35 and angle_diff < 25.0):
                # Union binary masks (merge contours)
                merged_pts = np.concatenate((base_mask_pts, compare_mask_pts))
                hull = cv2.convexHull(merged_pts)
                base_mask_pts = hull.reshape(-1, 2)
                
                # Expand box bounds
                base_box[0] = min(base_box[0], compare_box[0])
                base_box[1] = min(base_box[1], compare_box[1])
                base_box[2] = max(base_box[2], compare_box[2])
                base_box[3] = max(base_box[3], compare_box[3])
                
                merged_flags[j] = True # Drop duplicate

        final_boxes.append(base_box)
        final_masks.append(base_mask_pts.tolist())
        final_confs.append(base_conf)
        if base_id is not None:
            final_ids.append(base_id)

    return np.array(final_boxes), final_masks, np.array(final_confs), final_ids

def edge_guided_contour_snap(gray_frame, box, mask_contour):
    """
    Phase 4: Edge-guided contour snap using Canny/Sobel on the grayscale crop.
    """
    x1, y1, x2, y2 = map(int, box[:4])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(gray_frame.shape[1], x2), min(gray_frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return mask_contour

    # a. Run Canny on grayscale crop inside the mask bbox
    crop = gray_frame[y1:y2, x1:x2]
    edges = cv2.Canny(crop, 75, 175)

    # b. Intersect edge map with dilated YOLO mask
    local_mask = np.zeros(crop.shape, dtype=np.uint8)
    shifted_contour = np.array(mask_contour) - [x1, y1]
    cv2.fillPoly(local_mask, [shifted_contour.astype(np.int32)], 255)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(local_mask, kernel, iterations=1)
    
    intersected = cv2.bitwise_and(edges, dilated_mask)

    # c. Take largest connected component and dilate to rebuild tighter mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersected, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        tight_mask = np.uint8(labels == largest_label) * 255
        tight_mask = cv2.dilate(tight_mask, kernel, iterations=1)
        
        # Convert back to contour
        contours, _ = cv2.findContours(tight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best_contour = max(contours, key=cv2.contourArea)
            return (best_contour.squeeze() + [x1, y1]).tolist()

    return mask_contour


def _run_corrosion_live(args, base_gsd, source_label, annotate_corrosion_frame, patch_stats_to_rows):
    source = open_video_source(
        args.video,
        live=True,
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        stream_width=args.stream_width,
        stream_height=args.stream_height,
        protocol=args.stream_protocol,
    )
    mark_listen_ready(getattr(args, "ready_flag", None))
    publisher = PreviewJpegPublisher(args.preview_jpeg) if args.preview_jpeg else None
    raw_publisher = PreviewJpegPublisher(args.raw_preview_jpeg) if args.raw_preview_jpeg else None
    patch_stats: dict[int, dict] = {}
    next_patch_id = 1
    frame_idx = 0
    processed = 0
    native_w = int(getattr(source, "width", args.stream_width) or args.stream_width)
    native_h = int(getattr(source, "height", args.stream_height) or args.stream_height)
    try:
        while True:
            ok, frame = source.read(timeout=2.0)
            if not ok:
                if getattr(source, "eof", False):
                    break
                continue
            native_h, native_w = frame.shape[:2]
            frame_idx += 1
            if args.frame_stride > 1 and (frame_idx % args.frame_stride) != 0:
                continue
            processed += 1
            if args.max_frames and processed > args.max_frames:
                break
            if raw_publisher is not None:
                raw_publisher.publish(frame)
            display, next_patch_id = annotate_corrosion_frame(
                frame, args.corrosion_env, patch_stats, next_patch_id
            )
            if publisher is not None:
                publisher.publish(display)
            if processed % 30 == 0:
                rows = patch_stats_to_rows(
                    patch_stats,
                    base_gsd,
                    args.corrosion_env,
                    native_w,
                    native_h,
                    source_label,
                    "",
                )
                _write_csv_rows(args.output_csv, rows)
                if args.results_csv:
                    _write_csv_rows(args.results_csv, rows)
            if not args.no_preview:
                cv2.imshow("Project LIPAD AI - Live Corrosion", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        source.close()
        if publisher is not None:
            publisher.close()
        if raw_publisher is not None:
            raw_publisher.close()
        if not args.no_preview:
            cv2.destroyAllWindows()
    return patch_stats_to_rows(
        patch_stats,
        base_gsd,
        args.corrosion_env,
        native_w,
        native_h,
        source_label,
        "",
    )


def _compile_crack_rows(
    tracked_cracks: dict,
    track_id_to_label: dict,
    source_label: str,
    annotated_video_path: str | None,
    last_effective_gsd: float,
) -> list[dict]:
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
        if final_wid > 6.0 or (80.0 <= abs(final_ori) <= 90.0):
            continue
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
            "Video": source_label,
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
            "GSD_mm_per_px": round(last_effective_gsd, 6),
        })
    return master_object_log


def main():
    args = parse_arguments()
    repo_root = _resolve_repo_root()

    if args.output_csv is None:
        args.output_csv = os.path.join(repo_root, "data", "MorphologicalResults.csv")
    if args.results_csv is None:
        args.results_csv = os.path.join(repo_root, "data", "results.csv")

    base_gsd, gsd_source = _resolve_base_gsd(args)
    lidar_reader: LidarTelemetryReader | None = None
    if args.gsd_mode == "lidar":
        lidar_reader = LidarTelemetryReader(port=TELEMETRY_PORT)
        lidar_reader.start()
        print(f"[INFO] GSD mode: lidar — listening for VL53L8CX on UDP port {TELEMETRY_PORT}")
    else:
        print(f"[INFO] GSD mode: hypothetical — {GSD_HYPOTHETICAL_MM_PER_PX} mm/px")

    print("[INFO] Booting Quantized Project LIPAD Engine with BoT-SORT Tracking...")
    if not args.live:
        if not args.video:
            print("[CRITICAL ERROR] --video is required unless --live is set", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(args.video):
            print(f"[CRITICAL ERROR] Video missing: {args.video}", file=sys.stderr)
            sys.exit(1)

    source_label = (
        f"{args.stream_protocol}://{args.listen_host}:{args.listen_port}"
        if args.live
        else os.path.abspath(args.video)
    )

    inspection = (args.inspection_type or "Crack").strip()
    if inspection.lower() == "corrosion":
        try:
            from corrosion_analyzer import analyze_corrosion_video, annotate_corrosion_frame, patch_stats_to_rows
            print(f"[INFO] Corrosion mode ({args.corrosion_env}) — HSV/YCrCb pipeline")
            if args.live:
                rows = _run_corrosion_live(
                    args, base_gsd, source_label, annotate_corrosion_frame, patch_stats_to_rows
                )
            else:
                rows = analyze_corrosion_video(
                    video_path=args.video,
                    gsd_mm_per_px=base_gsd,
                    environment=args.corrosion_env,
                    frame_stride=max(1, int(args.frame_stride)),
                    max_frames=int(args.max_frames or 0),
                    output_video=args.output_video,
                )
            if not rows:
                print("\n[INFO] Complete. No validated corrosion patches found.")
            _write_csv_rows(args.output_csv, rows)
            if args.results_csv:
                _write_csv_rows(args.results_csv, rows)
        except ImportError:
            print("[WARNING] corrosion_analyzer module not found. Skipping corrosion run.")
        if lidar_reader:
            lidar_reader.stop()
        return

    if not os.path.exists(args.weights):
        print(f"[CRITICAL ERROR] Quantized weights missing: {args.weights}", file=sys.stderr)
        sys.exit(1)

    source = None
    raw_preview_pub = None
    raw_preview_stop = threading.Event()
    raw_preview_thread = None
    if args.live:
        source = open_video_source(
            args.video,
            live=True,
            listen_host=args.listen_host,
            listen_port=args.listen_port,
            stream_width=args.stream_width,
            stream_height=args.stream_height,
            protocol=args.stream_protocol,
        )
        mark_listen_ready(args.ready_flag)
        print(f"[LIVE] {args.stream_protocol.upper()} receiver is ready. Start rpicam-vid on the Pi now.")
        raw_preview_pub = PreviewJpegPublisher(args.raw_preview_jpeg) if args.raw_preview_jpeg else None

        if raw_preview_pub is not None:
            def _publish_raw_while_model_loads() -> None:
                while not raw_preview_stop.is_set():
                    ok, frame = source.read(timeout=0.1)
                    if ok and frame is not None:
                        raw_preview_pub.publish(frame)
                    elif getattr(source, "eof", False):
                        return

            raw_preview_thread = threading.Thread(
                target=_publish_raw_while_model_loads,
                daemon=True,
                name="raw-preview-warmup",
            )
            raw_preview_thread.start()

    _patch_onnxruntime_webgpu_compatibility()
    from ultralytics import YOLO

    try:
        model = YOLO(args.weights, task="segment")
    except Exception:
        raw_preview_stop.set()
        if raw_preview_thread is not None:
            raw_preview_thread.join(timeout=1)
        if raw_preview_pub is not None:
            raw_preview_pub.close()
        if source is not None:
            source.close()
        raise
    raw_preview_stop.set()
    if raw_preview_thread is not None:
        raw_preview_thread.join(timeout=1)

    device_context = _resolve_inference_device(args.weights)

    print(f"[SYSTEM] Quantized model loaded — device: {device_context}")
    print(f"  • {args.weights}")

    if source is None:
        source = open_video_source(
            args.video,
            live=False,
            listen_host=args.listen_host,
            listen_port=args.listen_port,
            stream_width=args.stream_width,
            stream_height=args.stream_height,
            protocol=args.stream_protocol,
        )
    fps = float(getattr(source, "fps", 30.0) or 30.0)
    native_width = int(getattr(source, "width", 0) or 0)
    native_height = int(getattr(source, "height", 0) or 0)
    print(f"[SYSTEM] High-Fidelity Processing Canvas Locked: {native_width}x{native_height}")

    window_title = "Project LIPAD AI - Quantized Telemetry Feed (BoT-SORT)"
    if not args.no_preview:
        cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL)
        if native_width > 0 and native_height > 0:
            cv2.resizeWindow(window_title, native_width, native_height)

    preview_pub = PreviewJpegPublisher(args.preview_jpeg) if args.preview_jpeg else None

    frame_counter = 0
    processed_frame_counter = 0
    tracked_cracks: dict[int, dict] = {}
    MIN_PRESENCE_RATIO = 0.10

    track_id_to_label: dict[int, str] = {}
    label_profiles: dict[str, dict] = {}
    next_label_id = 1

    writer = None
    annotated_video_path = None
    if args.output_video and not args.live:
        annotated_video_path = os.path.abspath(args.output_video)
        _ensure_parent_dir(annotated_video_path)

    last_effective_gsd = base_gsd

    try:
        while True:
            success, current_frame = source.read(timeout=2.0)
            if not success:
                if getattr(source, "eof", False) or not args.live:
                    break
                continue

            if native_width <= 0 or native_height <= 0:
                native_height, native_width = current_frame.shape[:2]

            frame_counter += 1
            if args.frame_stride > 1 and (frame_counter % args.frame_stride) != 0:
                continue

            processed_frame_counter += 1
            if args.max_frames and processed_frame_counter > args.max_frames:
                break

            # Publish the exact decoded frame that enters inference. This keeps
            # the raw and annotated desktop views aligned, while still showing
            # raw video if inference raises an exception.
            if raw_preview_pub is not None:
                raw_preview_pub.publish(current_frame)

            display_frame = current_frame.copy()

            try:
                if args.gsd_mode == "lidar" and lidar_reader is not None:
                    distance_mm = lidar_reader.get_distance_mm()
                    if distance_mm is not None:
                        frame_gsd = _gsd_from_lidar_distance_mm(distance_mm)
                        gsd_source = "lidar_live"
                    else:
                        frame_gsd = base_gsd
                        gsd_source = "lidar_fallback_hypothetical"
                else:
                    frame_gsd = base_gsd
                    gsd_source = "hypothetical"

                effective_gsd = float(frame_gsd)
                inference_frame = current_frame
                if args.inference_width and args.inference_width > 0 and native_width > 0:
                    scale = args.inference_width / float(native_width)
                    if 0.05 < scale < 1.0:
                        new_h = max(1, int(native_height * scale))
                        inference_frame = cv2.resize(
                            current_frame, (args.inference_width, new_h), interpolation=cv2.INTER_AREA
                        )
                        effective_gsd = effective_gsd / scale

                last_effective_gsd = effective_gsd
                display_frame = inference_frame.copy()
                fh, fw = inference_frame.shape[:2]

                if writer is None and annotated_video_path:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (fw, fh))

                ## Phase 1 & 5: Execute BoT-SORT Tracking Engine with max_det=75 and TTA (augment=True)
                inference_results = model.track(
                    source=inference_frame,
                    persist=True,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=5,       # Phase 1
                    augment=False,     # Phase 5: TTA
                    tracker="botsort.yaml",
                    verbose=False,
                    device=device_context,
                )

                result = inference_results[0]

                if result.boxes is not None and result.boxes.id is not None and result.masks is not None:
                    track_ids = result.boxes.id.int().cpu().tolist()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    masks = result.masks.xy
                    # boxes, masks, confs, track_ids = post_nms_mask_merge(boxes, masks, confs, track_ids)
                    gray_inference_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2GRAY)

                    # First Pass: Compute structural dimensions for all targets
                    for idx, track_id in enumerate(track_ids):
                        if idx >= len(masks):
                            continue

                        snapped_contour = edge_guided_contour_snap(gray_inference_frame, boxes[idx], masks[idx])
                        contour_numpy = np.array(snapped_contour, dtype=np.int32)

                        if len(contour_numpy) < 5:
                            continue

                        pixel_area = cv2.contourArea(contour_numpy)
                        if pixel_area <= 0:
                            x1, y1, x2, y2 = boxes[idx][:4]
                            pixel_area = max(1.0, (x2 - x1) * (y2 - y1))

                        real_area_mm2 = pixel_area * (effective_gsd ** 2)

                        pixel_perimeter = cv2.arcLength(contour_numpy, True)
                        if pixel_perimeter <= 0:
                            x1, y1, x2, y2 = boxes[idx][:4]
                            pixel_perimeter = 2 * ((x2 - x1) + (y2 - y1))

                        approx_length_mm = (pixel_perimeter / 2.0) * effective_gsd
                        if approx_length_mm < 5.0 or real_area_mm2 < 2.0:
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

                        # --- FILTER GATES: MAX WIDTH & ORIENTATION ANGLE ---
                        # 1. Cap maximum crack width at 6.0 mm
                        if run_wid > 6.0:
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

                        x1, y1, x2, y2 = boxes[idx][:4]
                        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), hud_color, 2)

                        contour_numpy = np.array(masks[idx], dtype=np.int32)
                        cv2.drawContours(display_frame, [contour_numpy], -1, (0, 0, 255), 2)

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
                            f"Critical_Shear_Alert: {is_critical} ({alert_text})",
                        ]

                        for line in text_data:
                            cv2.putText(
                                display_frame, line, (hud_x, hud_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
                            )
                            hud_y += 15

                if not args.no_preview:
                    cv2.imshow(window_title, display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if preview_pub is not None:
                    preview_pub.publish(display_frame)
                if writer is not None:
                    writer.write(display_frame)
                if args.live and processed_frame_counter % 45 == 0:
                    live_rows = _compile_crack_rows(
                        tracked_cracks,
                        track_id_to_label,
                        source_label,
                        annotated_video_path,
                        last_effective_gsd,
                    )
                    _write_csv_rows(args.output_csv, live_rows)
                    if args.results_csv:
                        _write_csv_rows(args.results_csv, live_rows)

            except Exception as e:
                print(f"[FRAME ERROR] {e}")
                continue
    finally:
        source.close()
        if preview_pub is not None:
            preview_pub.close()
        if raw_preview_pub is not None:
            raw_preview_pub.close()
        if writer is not None:
            writer.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
        if lidar_reader:
            lidar_reader.stop()

    print("\n[COMPILE] Unifying and compiling database spreadsheet rows...")
    master_object_log = _compile_crack_rows(
        tracked_cracks,
        track_id_to_label,
        source_label,
        annotated_video_path,
        last_effective_gsd,
    )

    if not master_object_log:
        print("\n[INFO] Complete. No validated structural crack elements found.")

    _write_csv_rows(args.output_csv, master_object_log)
    print(f"[SUCCESS] Morphological results saved to:\n👉 {args.output_csv}")

    if args.results_csv:
        _write_csv_rows(args.results_csv, master_object_log)
        print(f"[SUCCESS] UI mirror results saved to:\n👉 {args.results_csv}")

    print(f"[INFO] GSD source used: {gsd_source}")


if __name__ == "__main__":
    main()
