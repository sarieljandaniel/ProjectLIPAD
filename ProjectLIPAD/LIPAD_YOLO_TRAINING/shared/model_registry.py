"""YOLOv8 / YOLOv11 / YOLOv12 / YOLOv26 segmentation model presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelFamily = Literal["yolov8", "yolov11", "yolov12", "yolov26"]
TaskType = Literal["crack", "corrosion"]


@dataclass(frozen=True)
class ModelSpec:
    family: ModelFamily
    weights: str
    description: str


# Ultralytics pretrained segmentation checkpoints (nano — swap n→s/m/l/x as needed)
CRACK_MODELS: dict[ModelFamily, ModelSpec] = {
    "yolov8": ModelSpec("yolov8", "yolov8m-seg.pt", "YOLOv8 medium segmentation"),
    "yolov11": ModelSpec("yolov11", "yolo11n-seg.pt", "YOLOv11 nano segmentation"),
    "yolov12": ModelSpec("yolov12", "yolo12n-seg.yaml", "YOLOv12 nano segmentation (architecture; no seg .pt in Ultralytics 8.4+)"),
    "yolov26": ModelSpec("yolov26", "yolo26n-seg.pt", "YOLOv26 nano segmentation (pretrained)"),
}

CORROSION_MODELS: dict[ModelFamily, ModelSpec] = {
    "yolov8": ModelSpec("yolov8", "yolov8m-seg.pt", "YOLOv8 medium segmentation (corrosion)"),
    "yolov11": ModelSpec("yolov11", "yolo11n-seg.pt", "YOLOv11 nano segmentation (corrosion)"),
    "yolov12": ModelSpec("yolov12", "yolo12n-seg.yaml", "YOLOv12 nano segmentation (architecture; no seg .pt in Ultralytics 8.4+)"),
    "yolov26": ModelSpec("yolov26", "yolo26n-seg.pt", "YOLOv26 nano segmentation (pretrained, corrosion)"),
}


def get_model_spec(task: TaskType, family: ModelFamily) -> ModelSpec:
    table = CRACK_MODELS if task == "crack" else CORROSION_MODELS
    if family not in table:
        raise ValueError(f"Unknown model family: {family}")
    return table[family]


def all_families() -> list[ModelFamily]:
    return ["yolov8", "yolov11", "yolov12", "yolov26"]
