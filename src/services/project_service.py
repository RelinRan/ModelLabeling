from __future__ import annotations

import json
from pathlib import Path

from src.models.annotation import Annotation
from src.models.project import ProjectSettings
from .annotation_service import AnnotationService
from .dataset_detector import DatasetDetector
from .label_group_store import LabelGroupStore


class ProjectService:
    def __init__(self, annotation_service: AnnotationService | None = None) -> None:
        self.annotation_service = annotation_service or AnnotationService()

    def save_settings(self, path: Path, settings: ProjectSettings) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        payload = settings.to_dict()
        # Label templates are application-level data and live in SQLite. Do
        # not create a second project-local copy that can drift or follow a
        # dataset accidentally.
        payload.pop("label_groups", None)
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def load_settings(self, path: Path) -> ProjectSettings:
        if not path.exists():
            settings = ProjectSettings()
        else:
            settings = ProjectSettings.from_dict(json.loads(path.read_text(encoding="utf-8")))
        settings.label_groups = LabelGroupStore().load_or_initialize(settings.label_groups)
        return settings

    @staticmethod
    def _path_candidates(root: Path, annotation_format: str) -> tuple[tuple[Path, Path], ...]:
        root = Path(root)
        format_name = annotation_format.lower()
        if format_name == "yolo":
            return (
                (root / "images", root / "labels"),
                (root / "train" / "images", root / "train" / "labels"),
                (root / "val" / "images", root / "val" / "labels"),
                (root, root / "labels"),
                (root, root),
            )
        if format_name == "coco":
            return (
                (root / "images", root / "annotations"),
                (root, root / "annotations"),
            )
        return (
            (root / "JPEGImages", root / "Annotations"),
            (root / "images", root / "Annotations"),
            (root, root / "Annotations"),
            (root, root),
        )

    @classmethod
    def detect_dataset_format(cls, root: Path) -> tuple[str, Path, Path]:
        detected = DatasetDetector.detect(Path(root))
        return detected.format_name, detected.image_dir, detected.annotation_dir

    @staticmethod
    def resolve_dataset_paths(root: Path, annotation_format: str) -> tuple[Path, Path]:
        root = Path(root)
        format_name = annotation_format.lower()
        candidates = ProjectService._path_candidates(root, format_name)
        for image_dir, annotation_dir in candidates:
            if image_dir.is_dir() and annotation_dir.is_dir():
                return image_dir, annotation_dir
        if root.is_dir():
            sibling_name = "labels" if format_name == "yolo" else "annotations" if format_name == "coco" else "Annotations"
            sibling = root.parent / sibling_name
            if sibling.is_dir():
                return root, sibling
        raise ValueError(f"未识别的{format_name.upper()}数据集目录: {root}")

    def save_current(
        self,
        project_path: Path,
        image_path: Path,
        annotations: list[Annotation],
        settings: ProjectSettings,
    ) -> None:
        if settings.annotation_dir is None:
            raise ValueError("annotation directory is not configured")
        result = self.annotation_service.save(image_path, annotations, settings.annotation_dir, settings)
        if not result.ok:
            raise OSError(result.error or "failed to save annotations")
        metadata_path = settings.metadata_path or project_path.with_name("annotations.json")
        data = self.annotation_service.load_internal_metadata(metadata_path)
        data[image_path.name] = annotations
        self.annotation_service.save_internal_metadata(metadata_path, data)
        self.save_settings(project_path, settings)
