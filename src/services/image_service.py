from __future__ import annotations

from pathlib import Path
from typing import Callable
import heapq
import os

from PIL import Image

from src.models.project import ImageRecord, ProjectSettings
from .annotation_service import AnnotationService


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageService:
    def __init__(self, annotation_service: AnnotationService | None = None) -> None:
        self.annotation_service = annotation_service or AnnotationService()

    def scan(
        self,
        directory: Path,
        annotation_dir: Path,
        settings: ProjectSettings,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
        metadata_only: bool = False,
        limit: int | None = None,
        build_annotation_index: bool = True,
        batch_callback: Callable[[list[ImageRecord]], None] | None = None,
        batch_size: int = 100,
        sort_paths: bool = True,
    ) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        batch: list[ImageRecord] = []
        annotation_index = None if metadata_only or not build_annotation_index else self.annotation_service.build_index(annotation_dir, settings.annotation_format)
        if limit is not None:
            # Avoid sorting tens of thousands of paths when only the first
            # preview batch is needed. nsmallest preserves the same filename
            # ordering while reducing memory and sort overhead.
            preview_limit = max(0, int(limit))
            direct = [
                entry for entry in os.scandir(directory)
                if entry.is_file() and Path(entry.name).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ] if directory.is_dir() else []
            if len(direct) >= preview_limit:
                paths = [Path(entry.path) for entry in heapq.nsmallest(preview_limit, direct, key=lambda item: item.name.lower())]
            else:
                candidates = (item for item in directory.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
                paths = heapq.nsmallest(preview_limit, candidates, key=lambda item: item.name.lower())
        else:
            candidates = (item for item in directory.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
            if sort_paths:
                paths = sorted(candidates, key=lambda item: item.name.lower())
            else:
                paths = []
                for item in candidates:
                    if cancel_callback and cancel_callback():
                        break
                    paths.append(item)
        total = len(paths)
        progress_step = max(1, total // 100)
        for index, path in enumerate(paths, start=1):
            try:
                if metadata_only:
                    width = height = 0
                    file_format = path.suffix.lstrip(".").upper()
                    result = None
                    status = "pending"
                else:
                    with Image.open(path) as image:
                        width, height = image.size
                        file_format = image.format or path.suffix.lstrip(".").upper()
                    relative_parent = path.parent.relative_to(directory)
                    image_annotation_dir = annotation_dir if settings.annotation_format == "coco" else annotation_dir / relative_parent
                    result = self.annotation_service.load(path, image_annotation_dir, settings, annotation_index, (width, height))
                    status = "error" if result.error else ("labeled" if result.annotations else "unlabeled")
                record = ImageRecord(
                        path=path,
                        width=width,
                        height=height,
                        file_format=file_format,
                        file_size=path.stat().st_size,
                        annotations=result.annotations if result else [],
                        status=status,
                        error=result.error if result else None,
                        metadata_loaded=not metadata_only,
                    )
                records.append(record)
                batch.append(record)
            except (OSError, ValueError) as exc:
                record = ImageRecord(
                        path=path,
                        width=0,
                        height=0,
                        file_format=path.suffix.lstrip(".").upper(),
                        file_size=0,
                        status="error",
                        error=str(exc),
                    )
                records.append(record)
                batch.append(record)
            finally:
                if batch_callback and len(batch) >= max(1, int(batch_size)):
                    batch_callback(list(batch))
                    batch.clear()
                if progress_callback and (index == 1 or index == total or index % progress_step == 0):
                    progress_callback(index, total)
                if cancel_callback and cancel_callback():
                    break
        if batch_callback and batch:
            batch_callback(list(batch))
        return records

    def populate_annotations(
        self,
        records: list[ImageRecord],
        directory: Path,
        annotation_dir: Path,
        settings: ProjectSettings,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
        build_annotation_index: bool = True,
    ) -> None:
        """Fill annotation data into already-discovered records in the worker thread."""
        annotation_index = self.annotation_service.build_index(annotation_dir, settings.annotation_format) if build_annotation_index else None
        total = len(records)
        progress_step = max(1, total // 100)
        for index, record in enumerate(records, start=1):
            try:
                with Image.open(record.path) as image:
                    width, height = image.size
                    file_format = image.format or record.path.suffix.lstrip(".").upper()
                relative_parent = record.path.parent.relative_to(directory)
                image_annotation_dir = annotation_dir if settings.annotation_format == "coco" else annotation_dir / relative_parent
                result = self.annotation_service.load(record.path, image_annotation_dir, settings, annotation_index, (width, height))
                record.width, record.height, record.file_format = width, height, file_format
                record.annotations, record.error = result.annotations, result.error
                record.status = "error" if result.error else ("labeled" if result.annotations else "unlabeled")
                record.metadata_loaded = True
            except (OSError, ValueError) as exc:
                record.status, record.error, record.metadata_loaded = "error", str(exc), True
            if progress_callback and (index == 1 or index == total or index % progress_step == 0):
                progress_callback(index, total)
            if cancel_callback and cancel_callback():
                break

    @staticmethod
    def filter_records(records: list[ImageRecord], query: str = "", status: str = "all", label: str = "") -> list[ImageRecord]:
        normalized = query.casefold().strip()
        label_query = label.casefold().strip()
        return [
            record
            for record in records
            if (not normalized or normalized in record.path.name.casefold())
            and (status == "all" or record.status == status)
            and (not label_query or any(label_query == annotation.label.casefold() for annotation in record.annotations))
        ]
