from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from typing import Any

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor


class ShapeType(str, Enum):
    RECTANGLE = "rectangle"
    SQUARE = "square"
    POLYGON = "polygon"
    KEYPOINT = "keypoint"


def label_color(label: str) -> str:
    """Return one stable display color for a label name.

    Label groups intentionally contain names only. Deriving the color from
    the name keeps existing and newly drawn annotations visually consistent
    without storing color state in the label panel.
    """
    digest = hashlib.sha1(str(label).strip().encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") % 360
    return QColor.fromHsv(hue, 220, 245).name()


def _point_to_dict(point: QPointF) -> dict[str, float]:
    return {"x": float(point.x()), "y": float(point.y())}


def _point_from_dict(value: dict[str, Any]) -> QPointF:
    return QPointF(float(value["x"]), float(value["y"]))


@dataclass
class Keypoint:
    """One named keypoint in original-image coordinates."""

    name: str
    point: QPointF
    visibility: int = 2

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("keypoint name must not be empty")
        self.visibility = int(self.visibility)
        if self.visibility not in (0, 1, 2):
            raise ValueError("keypoint visibility must be 0, 1, or 2")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "point": _point_to_dict(self.point),
            "visibility": self.visibility,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Keypoint":
        return cls(
            name=value["name"],
            point=_point_from_dict(value["point"]),
            visibility=value.get("visibility", 2),
        )


@dataclass
class Annotation:
    shape_type: ShapeType
    label: str
    points: list[QPointF]
    color: str | None = None
    confidence: float | None = None
    source: str = "manual"
    keypoints: list[Keypoint] = field(default_factory=list)
    schema_name: str | None = None

    def __post_init__(self) -> None:
        self.shape_type = ShapeType(self.shape_type)
        self.label = str(self.label).strip()
        if not self.label:
            raise ValueError("label must not be empty")
        if self.shape_type != ShapeType.KEYPOINT and len(self.points) < 2:
            raise ValueError("annotation requires at least two points")
        if self.shape_type == ShapeType.KEYPOINT and not self.keypoints:
            raise ValueError("keypoint annotation requires at least one keypoint")
        if self.shape_type == ShapeType.KEYPOINT and len(self.points) not in (0, 2):
            raise ValueError("keypoint annotation points must be an optional bbox")
        self.color = label_color(self.label) if not self.color else str(self.color)
        if self.confidence is not None:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape_type": self.shape_type.value,
            "label": self.label,
            "points": [_point_to_dict(point) for point in self.points],
            "color": self.color,
            "confidence": self.confidence,
            "source": self.source,
            "keypoints": [item.to_dict() for item in self.keypoints],
            "schema_name": self.schema_name,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Annotation":
        return cls(
            shape_type=ShapeType(value["shape_type"]),
            label=value["label"],
            points=[_point_from_dict(point) for point in value["points"]],
            color=label_color(value["label"]),
            confidence=value.get("confidence"),
            source=value.get("source", "manual"),
            keypoints=[Keypoint.from_dict(item) for item in value.get("keypoints", [])],
            schema_name=value.get("schema_name"),
        )


@dataclass
class LabelPreset:
    name: str
    class_id: int
    color: str
    enabled: bool = True

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("label name must not be empty")
        if int(self.class_id) < 0:
            raise ValueError("class_id must be non-negative")
        self.class_id = int(self.class_id)
        self.color = str(self.color or "#00e5ff")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class_id": self.class_id,
            "color": self.color,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LabelPreset":
        return cls(
            name=value["name"],
            class_id=value["class_id"],
            color=value.get("color", "#00e5ff"),
            enabled=value.get("enabled", True),
        )
