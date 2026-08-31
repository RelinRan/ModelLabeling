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
    YOLO_OBB = "yolo_obb"
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
        DatasetTask.COCO, "COCO",
        # Standard COCO has no rotated box; OBB is YOLO-only here.
        frozenset(shape for shape in ShapeType if shape != ShapeType.OBB),
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
    DatasetTask.YOLO_OBB: FormatCapabilities(
        DatasetTask.YOLO_OBB, "YOLO OBB",
        # Ultralytics OBB rows are exactly four normalized corner points.
        frozenset({ShapeType.OBB}),
    ),
    DatasetTask.VOC: FormatCapabilities(
        DatasetTask.VOC, "Pascal VOC", frozenset({ShapeType.RECTANGLE, ShapeType.SQUARE}),
    ),
}




def _polygon_self_intersects(points) -> bool:
    """True when any two non-adjacent edges of the polygon cross."""
    n = len(points)
    for i in range(n):
        a1, a2 = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            b1, b2 = points[j], points[(j + 1) % n]
            if i == j or (j + 1) % n == i or (i + 1) % n == j:
                continue
            if _segments_cross(a1, a2, b1, b2):
                return True
    return False

def _segments_cross(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        value = (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x())
        return (value > 1e-9) - (value < -1e-9)
    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    return False
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
        "yolo_obb": DatasetTask.YOLO_OBB,
        "obb": DatasetTask.YOLO_OBB,
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
    if capabilities.task == DatasetTask.YOLO_SEGMENTATION:
        multipart = [
            annotation for annotation in annotations
            if annotation.shape_type == ShapeType.POLYGON
            and len(annotation.polygon_parts) > 1
        ]
        if multipart:
            raise UnsupportedAnnotationError(
                "YOLO Segmentation does not support multipart polygon instances"
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
    if ShapeType.POLYGON in {a.shape_type for a in annotations}:
        self_intersecting = [
            a.label for a in annotations
            if a.shape_type == ShapeType.POLYGON
            and len(a.points) >= 4
            and _polygon_self_intersects(a.points)
        ]
        if self_intersecting:
            raise UnsupportedAnnotationError(
                "多边形存在自交，请调整顶点: " + ", ".join(sorted(set(self_intersecting)))
            )

    if capabilities.task == DatasetTask.YOLO_OBB:
        bad = [annotation.label for annotation in annotations if len(annotation.points) != 4]
        if bad:
            raise UnsupportedAnnotationError(
                "YOLO OBB requires exactly four corner points: " + ", ".join(sorted(set(bad)))
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
