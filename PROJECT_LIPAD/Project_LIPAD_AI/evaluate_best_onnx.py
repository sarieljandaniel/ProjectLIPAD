"""Evaluate the exact quantized Project LiPAD ONNX checkpoint.

This script evaluates the deployed crack model, rather than using metrics from
an unrelated training run. It reports box and mask precision, recall, F1,
mAP@0.50, mAP@0.50:0.95, and preprocessing/inference/post-processing speed.

Example:
    python evaluate_best_onnx.py \
        --data path/to/eval_config.yaml \
        --weights "C:\\Users\\lenovo\\Project_LIPAD_V5\\ProjectLIPAD\\PROJECT_LIPAD\\models\\best.onnx"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

DEFAULT_WEIGHTS = r"C:\Users\lenovo\Project_LIPAD_V5\ProjectLIPAD\PROJECT_LIPAD\models\best.onnx"
DEFAULT_IMAGE_SIZE = 640


def f1_score(precision: float, recall: float) -> float:
    """Calculate the harmonic mean of precision and recall."""
    denominator = precision + recall
    return 0.0 if denominator == 0 else 2 * precision * recall / denominator


def metric_value(metrics: Any, name: str) -> float:
    """Read a numeric Ultralytics metric and normalize it to a float."""
    value = getattr(metrics, name, 0.0)
    return float(value) if value is not None else 0.0


def evaluate(weights: Path, data: Path, image_size: int, split: str) -> dict[str, Any]:
    """Evaluate the requested ONNX file on the requested dataset split."""
    if not weights.is_file():
        raise FileNotFoundError(f"ONNX weights not found: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")

    # Explicit task='segment' is required so ONNX mask metrics are evaluated.
    model = YOLO(str(weights), task="segment")
    validation = model.val(
        data=str(data),
        split=split,
        imgsz=image_size,
        plots=True,
        verbose=False,
        name=f"lipad_best_onnx_{split}",
    )

    box = validation.box
    mask = validation.seg
    result = {
        "weights": str(weights.resolve()),
        "data": str(data.resolve()),
        "split": split,
        "image_size": image_size,
        "save_dir": str(validation.save_dir),
        "box": {
            "precision": metric_value(box, "mp"),
            "recall": metric_value(box, "mr"),
            "map50": metric_value(box, "map50"),
            "map50_95": metric_value(box, "map"),
        },
        "mask": {
            "precision": metric_value(mask, "mp"),
            "recall": metric_value(mask, "mr"),
            "map50": metric_value(mask, "map50"),
            "map50_95": metric_value(mask, "map"),
        },
        "speed_ms_per_image": {
            "preprocess": float(validation.speed.get("preprocess", 0.0)),
            "inference": float(validation.speed.get("inference", 0.0)),
            "postprocess": float(validation.speed.get("postprocess", 0.0)),
        },
    }
    result["box"]["f1"] = f1_score(result["box"]["precision"], result["box"]["recall"])
    result["mask"]["f1"] = f1_score(result["mask"]["precision"], result["mask"]["recall"])
    return result


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    """Save machine-readable JSON and a compact CSV summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    rows = []
    for category in ("box", "mask"):
        for metric, value in result[category].items():
            rows.append({"category": category, "metric": metric, "value": value})
    for metric, value in result["speed_ms_per_image"].items():
        rows.append({"category": "speed_ms_per_image", "metric": metric, "value": value})

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def print_report(result: dict[str, Any]) -> None:
    """Print the metrics in a format suitable for copying into a report."""
    print("\n=== Project LiPAD exact ONNX evaluation ===")
    print(f"Weights: {result['weights']}")
    print(f"Dataset: {result['data']}")
    print(f"Split: {result['split']} | Image size: {result['image_size']}")
    print(f"Plots and validation files: {result['save_dir']}\n")
    print(f"{'Metric':<28}{'Box':>12}{'Mask':>12}")
    print("-" * 52)
    for metric in ("precision", "recall", "f1", "map50", "map50_95"):
        print(f"{metric:<28}{result['box'][metric]:>12.4f}{result['mask'][metric]:>12.4f}")
    print("\nSpeed (milliseconds per image)")
    for metric, value in result["speed_ms_per_image"].items():
        print(f"{metric:<28}{value:>12.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path(DEFAULT_WEIGHTS))
    parser.add_argument("--data", type=Path, required=True, help="Validation dataset YAML")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lipad_best_onnx_metrics"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args.weights, args.data, args.imgsz, args.split)
    write_outputs(result, args.output_dir)
    print_report(result)
    print(f"\nSaved summary: {args.output_dir / 'metrics.csv'}")
    print(f"Saved details: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
