from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from src.models.annotation import Annotation, ShapeType


class DatasetTask(str, Enum):
    COCO = "coco"
    YOLO_DETECTION = "yolo_detection"
    YOLO_SEGMENTATION = "yolo_segmentation"
    YOLO_POSE = "yolo_pose"
    VOC = "voc"


@dataclass(frozen=True)
class FormatCapabilities:
    task: DatasetTask
    display_name: str
    shapes: frozenset[ShapeType]

    def supports(self, shape_type: ShapeType) -> bool:
        return ShapeType(shape_type) in self.shapes


CAPABILITIES: dict[DatasetTask, FormatCapabilities] = {
    DatasetTask.COCO: FormatCapabilities(
        DatasetTask.COCO, "COCO", frozenset(ShapeType),
    ),
    DatasetTask.YOLO_DETECTION: FormatCapabilities(
        DatasetTask.YOLO_DETECTION, "YOLO Detection",
        frozenset({ShapeType.RECTANGLE, ShapeType.SQUARE}),
    ),
    DatasetTask.YOLO_SEGMENTATION: FormatCapabilities(
        DatasetTask.YOLO_SEGMENTATION, "YOLO Segmentation",
        frozenset({ShapeType.RECTANGLE, ShapeType.SQUARE, ShapeType.POLYGON}),
    ),
    DatasetTask.YOLO_POSE: FormatCapabilities(
        DatasetTask.YOLO_POSE, "YOLO Pose",
        # Official YOLO Pose rows require the keypoint triplets in addition
        # to the outer bbox; a bbox-only object is not a valid pose label.
        frozenset({ShapeType.KEYPOINT}),
    ),
    DatasetTask.VOC: FormatCapabilities(
        DatasetTask.VOC, "Pascal VOC", frozenset({ShapeType.RECTANGLE, ShapeType.SQUARE}),
    ),
}


class UnsupportedAnnotationError(ValueError):
    pass


def task_for_format(annotation_format: str, task: str | DatasetTask | None = None) -> DatasetTask:
    if isinstance(task, DatasetTask):
        return task
    normalized = str(task or "").strip().lower()
    aliases = {
        "coco": DatasetTask.COCO,
        "yolo": DatasetTask.YOLO_DETECTION,
        "yolo_detection": DatasetTask.YOLO_DETECTION,
        "yolo_segmentation": DatasetTask.YOLO_SEGMENTATION,
        "yolo_pose": DatasetTask.YOLO_POSE,
        "voc": DatasetTask.VOC,
        "pascal_voc": DatasetTask.VOC,
    }
    if normalized in aliases:
        return aliases[normalized]
    format_name = str(annotation_format).strip().lower()
    if format_name == "coco":
        return DatasetTask.COCO
    if format_name == "voc":
        return DatasetTask.VOC
    return DatasetTask.YOLO_DETECTION


def validate_annotations(
    annotations: Iterable[Annotation],
    annotation_format: str,
    task: str | DatasetTask | None = None,
) -> FormatCapabilities:
    capabilities = CAPABILITIES[task_for_format(annotation_format, task)]
    unsupported = sorted({
        annotation.shape_type.value
        for annotation in annotations
        if not capabilities.supports(annotation.shape_type)
    })
    if unsupported:
        raise UnsupportedAnnotationError(
            f"{capabilities.display_name} does not support: {', '.join(unsupported)}"
        )
    if capabilities.task == DatasetTask.YOLO_POSE:
        counts = {len(annotation.keypoints) for annotation in annotations}
        if 0 in counts:
            raise UnsupportedAnnotationError("YOLO Pose requires at least one keypoint per annotation")
        if len(counts) > 1:
            raise UnsupportedAnnotationError(
                "YOLO Pose requires one consistent keypoint count across the dataset"
            )
        schemas = {
            tuple(keypoint.name for keypoint in annotation.keypoints)
            for annotation in annotations
        }
        if len(schemas) > 1:
            raise UnsupportedAnnotationError(
                "YOLO Pose requires one consistent keypoint schema across the dataset"
            )
    if capabilities.task == DatasetTask.COCO:
        schemas_by_label: dict[str, set[tuple[str, ...]]] = {}
        for annotation in annotations:
            if annotation.keypoints:
                schemas_by_label.setdefault(annotation.label, set()).add(
                    tuple(keypoint.name for keypoint in annotation.keypoints)
                )
        inconsistent = sorted(
            label for label, schemas in schemas_by_label.items() if len(schemas) > 1
        )
        if inconsistent:
            raise UnsupportedAnnotationError(
                "COCO requires one keypoint schema per category: " + ", ".join(inconsistent)
            )
    return capabilities
