from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from src.models.annotation import Annotation, LabelPreset, ShapeType
from src.models.project import ProjectSettings
from .format_capabilities import CAPABILITIES, DatasetTask, task_for_format

if TYPE_CHECKING:
    from .annotation_service import AnnotationService, LoadResult, SaveResult


class DatasetFormatAdapter(Protocol):
    format_name: str
    task: DatasetTask

    def load(self, service: "AnnotationService", image: Path, directory: Path, settings: ProjectSettings):
        ...

    def save(self, service: "AnnotationService", image: Path, annotations: list[Annotation], directory: Path, settings: ProjectSettings):
        ...


@dataclass(frozen=True)
class StandardFormatAdapter:
    format_name: str
    task: DatasetTask

    @property
    def shapes(self) -> frozenset[ShapeType]:
        return CAPABILITIES[self.task].shapes

    def load(self, service: "AnnotationService", image: Path, directory: Path, settings: ProjectSettings, index=None, image_size=None):
        return service._load_by_adapter(image, directory, settings, self, index, image_size)

    def save(self, service: "AnnotationService", image: Path, annotations: list[Annotation], directory: Path, settings: ProjectSettings):
        return service._save_by_adapter(image, annotations, directory, settings, self)


def adapter_for(annotation_format: str, task: str | DatasetTask | None = None) -> StandardFormatAdapter:
    selected = task_for_format(annotation_format, task)
    return StandardFormatAdapter(selected.value, selected)


ADAPTERS: dict[str, StandardFormatAdapter] = {
    task.value: StandardFormatAdapter(task.value, task)
    for task in DatasetTask
}
