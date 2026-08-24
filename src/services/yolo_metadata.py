from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def find_yolo_yaml(root: Path) -> Path | None:
    current = Path(root)
    candidates: list[Path] = []
    for parent in (current, *list(current.parents)[:3]):
        candidates.extend((parent / "data.yaml", parent / "data.yml"))
    return next((path for path in candidates if path.is_file()), None)


def load_yolo_metadata(root: Path) -> dict[str, Any]:
    path = find_yolo_yaml(root)
    if path is None:
        return {}
    document = yaml.safe_load(path.read_text(encoding="utf-8", errors="strict"))
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ValueError(f"YOLO metadata must be a mapping: {path}")
    return document


def yolo_class_names(root: Path) -> list[str]:
    names = load_yolo_metadata(root).get("names", [])
    if isinstance(names, list):
        return [str(name).strip() for name in names]
    if isinstance(names, dict):
        indexed = {int(class_id): str(name).strip() for class_id, name in names.items()}
        return [indexed.get(index, f"class_{index}") for index in range(max(indexed, default=-1) + 1)]
    raise ValueError("YOLO names must be a list or class-id mapping")


def yolo_keypoint_shape(root: Path) -> tuple[int, int] | None:
    shape = load_yolo_metadata(root).get("kpt_shape")
    if shape is None:
        return None
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise ValueError("YOLO Pose kpt_shape must be [count, dimensions]")
    return int(shape[0]), int(shape[1])


def yolo_keypoint_names(root: Path, class_id: int = 0) -> list[str]:
    value = load_yolo_metadata(root).get("kpt_names", [])
    if isinstance(value, dict):
        value = value.get(class_id, value.get(str(class_id), []))
    elif isinstance(value, list) and value and isinstance(value[0], list):
        value = value[class_id] if class_id < len(value) else []
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise ValueError("YOLO Pose kpt_names must be a list or class-id mapping")
    return [str(name).strip() for name in value]
