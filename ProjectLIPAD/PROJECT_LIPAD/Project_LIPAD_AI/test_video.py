import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Test video inference using PyTorch quantized/best.pt model")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt model weights")
    parser.add_argument("--video", type=str, required=True, help="Path to input MP4 video file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--inference_width", type=int, default=640, help="Resize width for faster processing (default: 640)")
    parser.add_argument("--frame_stride", type=int, default=1, help="Process every Nth frame (default: 1)")
    parser.add_argument("--output_video", type=str, default=None, help="Optional path to save annotated output video")
    parser.add_argument("--no_preview", action="store_true", help="Disable live OpenCV display window")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Validate File Paths
    if not os.path.exists(args.weights):
        print(f"❌ [ERROR] Weights file not found: {args.weights}")
        sys.exit(1)
    if not os.path.exists(args.video):
        print(f"❌ [ERROR] Video file not found: {args.video}")
        sys.exit(1)

    # 2. Select Device (GPU / CPU)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🚀 [SYSTEM] Loading model onto device: '{device}'")
    if device == 0:
        print(f"   • GPU Detected: {torch.cuda.get_device_name(0)}")

    # 3. Load YOLO Model
    model = YOLO(args.weights)

    # 4. Initialize Video Capture
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 [VIDEO] Input resolution: {orig_w}x{orig_h} @ {fps:.1f} FPS")

    # 5. Initialize Optional Video Writer
    writer = None
    if args.output_video:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Target export resolution based on inference width scaling
        scale = args.inference_width / float(orig_w) if args.inference_width > 0 else 1.0
        out_w = args.inference_width if args.inference_width > 0 else orig_w
        out_h = int(orig_h * scale) if args.inference_width > 0 else orig_h
        writer = cv2.VideoWriter(args.output_video, fourcc, fps, (out_w, out_h))
        print(f"💾 [OUTPUT] Saving annotated video to: {args.output_video}")

    window_name = "PyTorch Model Test Feed"
    if not args.no_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if args.frame_stride > 1 and (frame_count % args.frame_stride) != 0:
            continue

        # Resize frame if requested
        if args.inference_width > 0 and orig_w > 0:
            scale = args.inference_width / float(orig_w)
            target_h = int(orig_h * scale)
            proc_frame = cv2.resize(frame, (args.inference_width, target_h), interpolation=cv2.INTER_AREA)
        else:
            proc_frame = frame

        # Run Inference
        results = model.predict(
            source=proc_frame,
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=False
        )[0]

        annotated_frame = proc_frame.copy()

        # Parse Detections & Render Overlay
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for idx, (box, conf) in enumerate(zip(boxes, confs)):
                x1, y1, x2, y2 = map(int, box[:4])

                # Draw Bounding Box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Crack: {conf:.2f}"
                cv2.putText(
                    annotated_frame, label, (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

                # Draw Mask Contour if present
                if results.masks is not None and idx < len(results.masks.xy):
                    polygon = np.array(results.masks.xy[idx], dtype=np.int32)
                    if len(polygon) >= 3:
                        cv2.polylines(annotated_frame, [polygon], True, (0, 0, 255), 2)

        # FPS Overlay
        elapsed = time.time() - start_time
        calc_fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(
            annotated_frame, f"FPS: {calc_fps:.1f} | Frame: {frame_count}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # Display and Write Frame
        if not args.no_preview:
            cv2.imshow(window_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⏹️ [INFO] User stopped video preview.")
                break

        if writer is not None:
            writer.write(annotated_frame)

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_preview:
        cv2.destroyAllWindows()

    print(f"✅ [FINISHED] Processed {frame_count} frames in {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    main()