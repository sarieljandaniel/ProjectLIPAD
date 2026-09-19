from ultralytics import YOLO

WEIGHTS = r"C:\Users\lenovo\Project_LIPAD_V5\ProjectLIPAD\PROJECT_LIPAD\models\best.onnx"
DATA = r"C:\Users\lenovo\Project_LIPAD_V5\ProjectLIPAD\PROJECT_LIPAD\Project_LIPAD_AI\dataset.yaml"  # adjust to your local path if needed

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