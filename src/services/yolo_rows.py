"""Strict parsing of the four standard Ultralytics annotation row layouts."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


class YoloRowError(ValueError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class YoloRow:
    class_id: int
    values: tuple[float, ...]
    box: tuple[float, float, float, float]


def parse_yolo_row(raw: str, task: str, kpt_shape: tuple[int, int] | None = None) -> YoloRow:
    parts = raw.split()
    count = len(parts)
    dimensions = kpt_shape[1] if kpt_shape else 3
    valid = {
        "yolo_detection": count == 5,
        "yolo_segmentation": count >= 7 and (count - 1) % 2 == 0,
        "yolo_obb": count == 9,
        "yolo_pose": count >= 5 + dimensions and (count - 5) % dimensions == 0,
    }
    if task not in valid or not valid[task]:
        raise YoloRowError("bad_arity", f"invalid {task} row: {count} columns")
    if task == "yolo_pose" and kpt_shape and count != 5 + kpt_shape[0] * dimensions:
        raise YoloRowError("bad_arity", "keypoint columns do not match kpt_shape")
    try:
        class_id = int(parts[0])
        values = tuple(float(value) for value in parts[1:])
    except ValueError as exc:
        raise YoloRowError("bad_number", "invalid class id or coordinate") from exc
    if not all(isfinite(value) for value in values):
        raise YoloRowError("bad_number", "coordinates must be finite")
    coordinates = values
    if task == "yolo_pose" and dimensions == 3:
        coordinates = values[:4] + tuple(value for i, value in enumerate(values[4:]) if i % 3 != 2)
        if any(value not in (0, 1, 2) for value in values[6::3]):
            raise YoloRowError("out_of_range", "visibility must be 0, 1, or 2")
    if any(value < -1e-6 or value > 1 + 1e-6 for value in coordinates):
        raise YoloRowError("out_of_range", "normalized coordinates must be in [0, 1]")
    if task in {"yolo_segmentation", "yolo_obb"}:
        points = list(zip(values[::2], values[1::2]))
        area = abs(sum(x * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * y
                       for i, (x, y) in enumerate(points))) / 2
        if area <= 1e-12:
            raise YoloRowError("degenerate", "polygon must have positive area")
        xs, ys = values[::2], values[1::2]
        box = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys))
    else:
        box = values[:4]
    if box[2] <= 0 or box[3] <= 0:
        raise YoloRowError("degenerate", "bbox dimensions must be positive")
    return YoloRow(class_id, values, box)
