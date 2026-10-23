from __future__ import annotations

from src.services.format_capabilities import DatasetTask

# Dataset format and task names are technical identifiers: they keep the same
# canonical English spelling in every UI language, so a dataset is called
# "Pascal VOC" everywhere instead of "VOC" in one dialog and
# "Pascal VOC（矩形框）" in another. Only the surrounding prose is translated.
FORMAT_NAMES = {
    "yolo": "YOLO",
    "voc": "Pascal VOC",
    "coco": "COCO",
}

TASK_NAMES = {
    DatasetTask.YOLO_DETECTION.value: "Detection",
    DatasetTask.YOLO_SEGMENTATION.value: "Segmentation",
    DatasetTask.YOLO_POSE.value: "Pose",
    DatasetTask.YOLO_OBB.value: "OBB",
    DatasetTask.VOC.value: "Pascal VOC",
    DatasetTask.COCO.value: "COCO",
}

#: Format dropdown order shared by every dialog.
FORMAT_VALUES = ("coco", "yolo", "voc")

#: New-dataset and task dropdown order (YOLO tasks first).
TASK_VALUES = (
    DatasetTask.YOLO_DETECTION.value,
    DatasetTask.YOLO_SEGMENTATION.value,
    DatasetTask.YOLO_POSE.value,
    DatasetTask.YOLO_OBB.value,
    DatasetTask.VOC.value,
    DatasetTask.COCO.value,
)


def _key(value: str | DatasetTask | None) -> str:
    if isinstance(value, DatasetTask):
        return value.value
    return str(value or "").strip().lower()


def format_name(value: str | DatasetTask | None, language: str = "zh_CN") -> str:
    key = _key(value)
    return FORMAT_NAMES.get(key, key.upper())


def task_name(value: str | DatasetTask | None, language: str = "zh_CN") -> str:
    key = _key(value)
    return TASK_NAMES.get(key, key)


def task_label(value: str | DatasetTask | None, language: str = "zh_CN") -> str:
    """Dropdown label: ``YOLO Detection``, ``Pascal VOC``, ``COCO``."""
    key = _key(value)
    if key.startswith("yolo_"):
        return f"{FORMAT_NAMES['yolo']} {task_name(key, language)}"
    if key in FORMAT_NAMES:
        return format_name(key, language)
    return task_name(key, language)


def format_task_name(format_value: str | DatasetTask | None, task_value: str | DatasetTask | None, language: str = "zh_CN") -> str:
    """Status-bar label: ``YOLO · Detection``, ``Pascal VOC``, ``COCO``."""
    task_key = _key(task_value)
    if task_key.startswith("yolo_"):
        return f"{FORMAT_NAMES['yolo']} · {task_name(task_key, language)}"
    if task_key in FORMAT_NAMES:
        return format_name(task_key, language)
    return format_name(format_value, language)


def format_options(language: str = "zh_CN") -> list[tuple[str, str]]:
    """``(label, value)`` pairs for a format dropdown."""
    return [(format_name(value, language), value) for value in FORMAT_VALUES]


def task_options(format_value: str | DatasetTask | None, language: str = "zh_CN") -> list[tuple[str, str]]:
    """``(label, value)`` pairs for the task dropdown of one format."""
    key = _key(format_value)
    if key == "yolo":
        return [(task_label(value, language), value) for value in TASK_VALUES if value.startswith("yolo_")]
    if key in ("voc", "coco"):
        return [(format_name(key, language), key)]
    return []
