from __future__ import annotations

from src.services.format_capabilities import DatasetTask


FORMAT_NAMES = {
    "zh_CN": {"yolo": "YOLO", "voc": "Pascal VOC", "coco": "COCO"},
    "en_US": {"yolo": "YOLO", "voc": "Pascal VOC", "coco": "COCO"},
}

TASK_NAMES = {
    "zh_CN": {
        DatasetTask.YOLO_DETECTION.value: "目标检测",
        DatasetTask.YOLO_SEGMENTATION.value: "实例分割",
        DatasetTask.YOLO_POSE.value: "关键点",
        DatasetTask.YOLO_OBB.value: "旋转目标检测",
        DatasetTask.VOC.value: "目标检测",
        DatasetTask.COCO.value: "检测 / 分割 / 关键点",
    },
    "en_US": {
        DatasetTask.YOLO_DETECTION.value: "Object Detection",
        DatasetTask.YOLO_SEGMENTATION.value: "Instance Segmentation",
        DatasetTask.YOLO_POSE.value: "Pose Keypoints",
        DatasetTask.YOLO_OBB.value: "Oriented Detection",
        DatasetTask.VOC.value: "Object Detection",
        DatasetTask.COCO.value: "Detection / Segmentation / Pose",
    },
}


def format_name(value: str | None, language: str = "zh_CN") -> str:
    key = str(value or "").strip().lower()
    return FORMAT_NAMES.get(language, FORMAT_NAMES["zh_CN"]).get(key, key.upper())


def task_name(value: str | DatasetTask | None, language: str = "zh_CN") -> str:
    key = value.value if isinstance(value, DatasetTask) else str(value or "").strip().lower()
    return TASK_NAMES.get(language, TASK_NAMES["zh_CN"]).get(key, key)


def format_task_name(format_value: str | None, task_value: str | DatasetTask | None, language: str = "zh_CN") -> str:
    return f"{format_name(format_value, language)} · {task_name(task_value or format_value, language)}"
