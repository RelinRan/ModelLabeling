from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from .annotation import Annotation, LabelPreset, ShapeType


SUPPORTED_FORMATS = {"yolo", "voc", "coco"}


@dataclass
class ImageRecord:
    path: Path
    width: int
    height: int
    file_format: str
    file_size: int
    annotations: list[Annotation] = field(default_factory=list)
    status: str = "unlabeled"
    error: str | None = None
    metadata_loaded: bool = True

    @property
    def is_labeled(self) -> bool:
        return bool(self.annotations)


@dataclass
class LabelGroup:
    name: str
    presets: list[LabelPreset] = field(default_factory=list)
    protected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "presets": [preset.to_dict() for preset in self.presets],
            "protected": self.protected,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LabelGroup":
        return cls(
            name=str(value.get("name", "Default")),
            presets=[LabelPreset.from_dict(item) for item in value.get("presets", [])],
            protected=bool(value.get("protected", False)),
        )


@dataclass
class ProjectSettings:
    image_dir: Path | None = None
    annotation_dir: Path | None = None
    annotation_format: str = "yolo"
    dataset_task: str | None = None
    label_presets: list[LabelPreset] = field(default_factory=list)
    line_width: int = 2
    text_size: int = 14
    crosshair_line_width: int = 2
    crosshair_color: str = "#ffea00"
    auto_save: bool = True
    onnx_model_path: Path | None = None
    input_size: int = 640
    input_width: int | None = None
    input_height: int | None = None
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45
    metadata_path: Path | None = None
    enabled_shapes: list[ShapeType] = field(
        default_factory=lambda: [ShapeType.RECTANGLE, ShapeType.SQUARE, ShapeType.POLYGON]
    )
    language: str = "zh_CN"
    label_groups: list[LabelGroup] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.annotation_format = self.annotation_format.lower()
        if self.annotation_format not in SUPPORTED_FORMATS:
            raise ValueError(f"unsupported annotation format: {self.annotation_format}")
        self.line_width = max(1, int(self.line_width))
        self.text_size = max(6, int(self.text_size))
        self.crosshair_line_width = max(1, min(12, int(self.crosshair_line_width)))
        if not isinstance(self.crosshair_color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", self.crosshair_color):
            self.crosshair_color = "#ffea00"
        self.input_size = max(32, int(self.input_size))
        self.input_width = max(32, int(self.input_width or self.input_size))
        self.input_height = max(32, int(self.input_height or self.input_size))
        self.confidence_threshold = max(0.0, min(1.0, float(self.confidence_threshold)))
        self.nms_threshold = max(0.0, min(1.0, float(self.nms_threshold)))
        self.enabled_shapes = list(dict.fromkeys(ShapeType(item) for item in self.enabled_shapes))
        if not self.enabled_shapes:
            self.enabled_shapes = [ShapeType.RECTANGLE]
        if self.language not in {"zh_CN", "en_US"}:
            self.language = "zh_CN"
        if not self.label_groups:
            self.label_groups = [LabelGroup("默认标签", list(self.label_presets), True)]
        if not self.label_presets:
            self.label_presets = list(self.label_groups[0].presets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_dir": str(self.image_dir) if self.image_dir else None,
            "annotation_dir": str(self.annotation_dir) if self.annotation_dir else None,
            "annotation_format": self.annotation_format,
            "dataset_task": self.dataset_task,
            "label_presets": [preset.to_dict() for preset in self.label_presets],
            "line_width": self.line_width,
            "text_size": self.text_size,
            "crosshair_line_width": self.crosshair_line_width,
            "crosshair_color": self.crosshair_color,
            "auto_save": self.auto_save,
            "onnx_model_path": str(self.onnx_model_path) if self.onnx_model_path else None,
            "input_size": self.input_size,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "enabled_shapes": [shape.value for shape in self.enabled_shapes],
            "language": self.language,
            "label_groups": [group.to_dict() for group in self.label_groups],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProjectSettings":
        def path_or_none(item: Any) -> Path | None:
            return Path(item) if item else None

        return cls(
            image_dir=path_or_none(value.get("image_dir")),
            annotation_dir=path_or_none(value.get("annotation_dir")),
            annotation_format=value.get("annotation_format", "yolo"),
            dataset_task=value.get("dataset_task"),
            label_presets=[LabelPreset.from_dict(item) for item in value.get("label_presets", [])],
            line_width=value.get("line_width", 2),
            text_size=value.get("text_size", 14),
            crosshair_line_width=value.get("crosshair_line_width", 2),
            crosshair_color=value.get("crosshair_color", "#ffea00"),
            auto_save=value.get("auto_save", True),
            onnx_model_path=path_or_none(value.get("onnx_model_path")),
            input_size=value.get("input_size", 640),
            input_width=value.get("input_width", value.get("input_size", 640)),
            input_height=value.get("input_height", value.get("input_size", 640)),
            confidence_threshold=value.get("confidence_threshold", 0.25),
            nms_threshold=value.get("nms_threshold", 0.45),
            metadata_path=path_or_none(value.get("metadata_path")),
            enabled_shapes=[ShapeType(item) for item in value.get(
                "enabled_shapes", [shape.value for shape in ShapeType]
            )],
            language=value.get("language", "zh_CN"),
            label_groups=[LabelGroup.from_dict(item) for item in value.get("label_groups", [])],
        )


@dataclass
class ProjectState:
    settings: ProjectSettings
    images: list[ImageRecord] = field(default_factory=list)
    current_index: int = -1

    @property
    def current_image(self) -> ImageRecord | None:
        if 0 <= self.current_index < len(self.images):
            return self.images[self.current_index]
        return None

    def statistics(self) -> dict[str, Any]:
        labels = Counter(
            annotation.label
            for image in self.images
            for annotation in image.annotations
        )
        labeled = sum(1 for image in self.images if image.is_labeled)
        total = len(self.images)
        return {
            "total_images": total,
            "labeled_images": labeled,
            "percentage": (labeled / total * 100.0) if total else 0.0,
            "total_labels": sum(labels.values()),
            "label_counts": dict(sorted(labels.items())),
        }
