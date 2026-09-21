"""Quantize ensemble crack-detection weights (YOLOv8s + YOLOv8m subsets).

Adapted from Sohaib et al. (Sensors 2024) — reduces weight/activation precision
before ensemble inference. Exports FP16 ONNX models for deployment in LiPAD.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]  # .../Corrosion
CRACK = REPO / "LIPAD_YOLO_TRAINING" / "crack_detection"
RUNS = CRACK / "runs" / "segment"
DEFAULT_OUT = REPO / "PROJECT_LIPAD" / "models" / "quantized"

VARIANTS = {
    "ensemble_subset_s": RUNS / "ensemble_subset_s" / "weights" / "best.pt",
    "ensemble_subset_m": RUNS / "ensemble_subset_m" / "weights" / "best.pt",
}


def quantize_variant(name: str, src: Path, out_dir: Path, *, imgsz: int = 640) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Missing trained weights: {src}")

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}_fp16.onnx"

    print(f"[quantize] {name}: loading {src}")
    model = YOLO(str(src))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        opset=12,
        half=True,  # FP16 — Ultralytics 8.x maps to quantize=16
    )
    Path(exported).replace(dest)
    print(f"[quantize] saved → {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize LiPAD ensemble YOLOv8 weights")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
    )
    args = parser.parse_args()

    for name in args.variants:
        quantize_variant(name, VARIANTS[name], args.out_dir, imgsz=args.imgsz)

    print("[quantize] Complete.")


if __name__ == "__main__":
    main()
