"""Evaluate the exact quantized Project LiPAD ONNX checkpoint.

The script evaluates the deployed ONNX model against both crack validation
configuration files supplied for the subset-m and subset-s datasets. It reports
box and mask precision, recall, F1, mAP@0.50, mAP@0.50:0.95, and speed.

Run from the ``PROJECT_LIPAD`` directory:

    python Project_LIPAD_AI/evaluate_best_onnx.py

<<<<<<< HEAD:PROJECT_LIPAD/Project_LIPAD_AI/evaluate_best_onnx.py
Optional arguments can override the default Windows paths.
=======
Weights and dataset YAMLs default to paths next to this repo, not a fixed drive.
Optional arguments can override those defaults.
>>>>>>> bd5c643273d6009ab0c3f5718e919bbc705e746a:ProjectLIPAD/PROJECT_LIPAD/Project_LIPAD_AI/evaluate_best_onnx.py
"""

from __future__ import annotations

import argparse
import csv
import json
<<<<<<< HEAD:PROJECT_LIPAD/Project_LIPAD_AI/evaluate_best_onnx.py
from pathlib import Path
from typing import Any

from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(
    r"C:\Users\lenovo\Project_LIPAD_V5\ProjectLIPAD\PROJECT_LIPAD\models\best.onnx"
)
DEFAULT_DATASETS = (
    Path(
        r"C:\Users\lenovo\Project_LIPAD_V5\ProjectLIPAD\LIPAD_YOLO_TRAINING\crack_detection\subset_m.yaml"
    ),
    Path(
        r"C:\Users\lenovo\Project_LIPAD_V5\ProjectLIPAD\LIPAD_YOLO_TRAINING\crack_detection\subset_s.yaml"
    ),
)
=======
import sys
from pathlib import Path
from typing import Any

_AI_DIR = Path(__file__).resolve().parent
_APP_ROOT = _AI_DIR.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from paths import DEFAULT_ONNX_WEIGHTS, SUBSET_M_YAML, SUBSET_S_YAML
from ultralytics import YOLO

DEFAULT_WEIGHTS = DEFAULT_ONNX_WEIGHTS
DEFAULT_DATASETS = (SUBSET_M_YAML, SUBSET_S_YAML)
>>>>>>> bd5c643273d6009ab0c3f5718e919bbc705e746a:ProjectLIPAD/PROJECT_LIPAD/Project_LIPAD_AI/evaluate_best_onnx.py
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
    """Evaluate the exact ONNX checkpoint against one dataset YAML."""
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
        name=f"lipad_best_onnx_{data.stem}_{split}",
    )

    box = validation.box
    mask = validation.seg
    result: dict[str, Any] = {
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


def safe_dataset_name(data: Path) -> str:
    """Return a filesystem-safe name for per-dataset output files."""
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in data.stem)


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    """Save machine-readable JSON and CSV summaries for one dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = safe_dataset_name(Path(result["data"]))
    (output_dir / f"{dataset_name}_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    rows = []
    for category in ("box", "mask"):
        for metric, value in result[category].items():
            rows.append({"category": category, "metric": metric, "value": value})
    for metric, value in result["speed_ms_per_image"].items():
        rows.append({"category": "speed_ms_per_image", "metric": metric, "value": value})

    with (output_dir / f"{dataset_name}_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_combined_csv(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Save one comparison CSV containing both subset evaluations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        dataset = Path(result["data"]).stem
        for category in ("box", "mask"):
            for metric, value in result[category].items():
                rows.append(
                    {"dataset": dataset, "category": category, "metric": metric, "value": value}
                )
        for metric, value in result["speed_ms_per_image"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "category": "speed_ms_per_image",
                    "metric": metric,
                    "value": value,
                }
            )

    with (output_dir / "combined_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "category", "metric", "value"]
        )
        writer.writeheader()
        writer.writerows(rows)


def print_report(result: dict[str, Any]) -> None:
    """Print metrics in a format suitable for an academic report."""
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
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="One or more validation dataset YAML files.",
    )
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/lipad_best_onnx_metrics")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for dataset in args.data:
        result = evaluate(args.weights, dataset, args.imgsz, args.split)
        results.append(result)
        write_outputs(result, args.output_dir)
        print_report(result)

    write_combined_csv(results, args.output_dir)
    combined_json = args.output_dir / "combined_metrics.json"
    combined_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved combined CSV: {args.output_dir / 'combined_metrics.csv'}")
    print(f"Saved combined JSON: {combined_json}")


if __name__ == "__main__":
    main()
