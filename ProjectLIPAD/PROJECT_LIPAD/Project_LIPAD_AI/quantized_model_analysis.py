from pathlib import Path
import sys

from ultralytics import YOLO

_AI_DIR = Path(__file__).resolve().parent
_APP_ROOT = _AI_DIR.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from paths import AI_DATASET_YAML, DEFAULT_ONNX_WEIGHTS

WEIGHTS = str(DEFAULT_ONNX_WEIGHTS)
DATA = str(AI_DATASET_YAML)

model = YOLO(WEIGHTS, task="segment")
metrics = model.val(
    data=DATA,
    split="val",
    imgsz=640,
    plots=True,
    verbose=False,
    name="best_onnx_eval"
)

box = metrics.box
mask = metrics.seg

print("Box Precision:", box.mp)
print("Box Recall:", box.mr)
print("Box mAP@0.5:", box.map50)
print("Box mAP@0.5:0.95:", box.map)

print("Mask Precision:", mask.mp)
print("Mask Recall:", mask.mr)
print("Mask mAP@0.5:", mask.map50)
print("Mask mAP@0.5:0.95:", mask.map)

print("Preprocess (ms/img):", metrics.speed["preprocess"])
print("Inference (ms/img):", metrics.speed["inference"])
print("Postprocess (ms/img):", metrics.speed["postprocess"])
