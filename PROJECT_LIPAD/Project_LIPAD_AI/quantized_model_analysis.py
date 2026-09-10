# Compare best.pt vs best.onnx on the same val split (eval_config.yaml).
from ultralytics import YOLO

PT_WEIGHTS = r"C:\Users\Admin\PROJECT_LIPAD\Corrosion\PROJECT_LIPAD\models\best.pt"
ONNX_WEIGHTS = r"C:\Users\lenovo\Project_LIPAD_v3\Corrosion\PROJECT_LIPAD\models\best.onnx"


def run_eval(weights: str, label: str, task: str | None = None):
    model = YOLO(weights, task=task) if task else YOLO(weights)
    metrics = model.val(
        data="eval_config.yaml",
        split="val",
        imgsz=640,
        plots=True,
        verbose=False,
        name=f"quantized_model_eval_{label.replace('.', '_')}",
    )
    box, mask = metrics.box, metrics.seg
    return {
        "label": label,
        "save_dir": str(metrics.save_dir),
        "box_p": box.mp,
        "box_r": box.mr,
        "box_map50": box.map50,
        "box_map5095": box.map,
        "mask_p": mask.mp,
        "mask_r": mask.mr,
        "mask_map50": mask.map50,
        "mask_map5095": mask.map,
        "preprocess": metrics.speed["preprocess"],
        "inference": metrics.speed["inference"],
        "postprocess": metrics.speed["postprocess"],
    }


def print_table(pt: dict, onnx: dict) -> None:
    rows = [
        ("Box Precision", "box_p"),
        ("Box Recall", "box_r"),
        ("Box mAP@0.5", "box_map50"),
        ("Box mAP@0.5:0.95", "box_map5095"),
        ("Mask Precision", "mask_p"),
        ("Mask Recall", "mask_r"),
        ("Mask mAP@0.5", "mask_map50"),
        ("Mask mAP@0.5:0.95", "mask_map5095"),
        ("Preprocess (ms/img)", "preprocess"),
        ("Inference (ms/img)", "inference"),
        ("Postprocess (ms/img)", "postprocess"),
    ]
    print("\n=== FP32/PT vs ONNX (eval_config.yaml, val, imgsz=640) ===")
    print(f"{'Metric':<22} {'best.pt':>10} {'best.onnx':>10} {'Delta':>10}")
    print("-" * 56)
    for name, key in rows:
        a, b = pt[key], onnx[key]
        d = b - a
        sign = "+" if d >= 0 else ""
        if "ms" in name:
            print(f"{name:<22} {a:>10.2f} {b:>10.2f} {sign}{d:>9.2f}")
        else:
            print(f"{name:<22} {a:>10.3f} {b:>10.3f} {sign}{d:>9.3f}")


if __name__ == "__main__":
    pt = run_eval(PT_WEIGHTS, "best.pt")
    # ONNX must be loaded with task='segment' or Ultralytics defaults to detect (metrics = 0).
    onnx = run_eval(ONNX_WEIGHTS, "best.onnx", task="segment")
    print_table(pt, onnx)
    print(f"\n[SUCCESS] PT results:   {pt['save_dir']}")
    print(f"[SUCCESS] ONNX results: {onnx['save_dir']}")
