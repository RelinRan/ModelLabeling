from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from typing import Callable

from PySide6.QtGui import QColor

from src.models.annotation import LabelPreset
from src.models.project import ProjectSettings
from .annotation_service import AnnotationService
from .dataset_detector import DatasetDetector
from src.models.keypoint import COCO_PERSON_KEYPOINTS


@dataclass
class ConversionOptions:
    source_format: str
    source_path: Path
    output_format: str
    output_path: Path
    presets: list
    overwrite: bool = False
    source_task: str | None = None
    output_task: str | None = None


@dataclass
class ConversionReport:
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


class ConversionService:
    def __init__(self, annotation_service: AnnotationService | None = None) -> None:
        self.annotation_service = annotation_service or AnnotationService()

    def convert(
        self,
        options: ConversionOptions,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> ConversionReport:
        report = ConversionReport()
        source_format = self._canonical_format(options.source_format)
        output_format = self._canonical_format(options.output_format)
        source_task = options.source_task or ("coco" if source_format == "coco" else "voc" if source_format == "voc" else "yolo_detection")
        output_task = options.output_task or ("coco" if output_format == "coco" else "voc" if output_format == "voc" else "yolo_detection")
        image_dir, annotation_dir, structured = self._resolve_source(options.source_path, source_format)
        images = sorted(
            item for item in image_dir.rglob("*")
            if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        presets = self._complete_presets(images, annotation_dir, source_format, options.presets, options.source_path)
        structured_output = structured or output_format == "coco"
        if structured_output:
            if output_format == "yolo":
                output_image_dir = options.output_path / "images"
                output_annotation_dir = options.output_path / "labels"
            elif output_format == "coco":
                output_image_dir = options.output_path / "images"
                output_annotation_dir = options.output_path / "annotations"
            else:
                output_image_dir = options.output_path / "JPEGImages"
                output_annotation_dir = options.output_path / "Annotations"
        else:
            output_image_dir = options.output_path
            output_annotation_dir = options.output_path
        if options.overwrite:
            # A COCO document is shared by every image. Remove stale or
            # interrupted output before the first image is written.
            if output_format == "coco":
                for path in (output_annotation_dir / "annotations.json", output_annotation_dir / "instances.json"):
                    path.unlink(missing_ok=True)
                for path in output_annotation_dir.glob(".model_labeling.sqlite3*"):
                    path.unlink(missing_ok=True)
                if output_image_dir.exists():
                    for path in output_image_dir.iterdir():
                        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                            path.unlink(missing_ok=True)
            elif output_format in {"voc", "yolo"}:
                suffixes = {"voc": {".xml"}, "yolo": {".txt"}}[output_format]
                for path in output_annotation_dir.iterdir() if output_annotation_dir.exists() else ():
                    if path.is_file() and path.suffix.lower() in suffixes:
                        path.unlink(missing_ok=True)
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_annotation_dir.mkdir(parents=True, exist_ok=True)
        if output_format == "yolo":
            self._write_yolo_classes(options.output_path, presets)
        settings = ProjectSettings(
            annotation_format=source_format,
            dataset_task=source_task,
            annotation_dir=annotation_dir,
            label_presets=presets,
        )
        if output_format == "coco":
            batch: list[tuple[Path, list]] = []
            for index, image_path in enumerate(images, start=1):
                if cancel_callback and cancel_callback():
                    break
                try:
                    result = self.annotation_service.load(image_path, annotation_dir, settings)
                    if result.error:
                        raise ValueError(result.error)
                    batch.append((image_path, result.annotations))
                    shutil.copy2(image_path, output_image_dir / image_path.name)
                    report.succeeded += 1
                except Exception as exc:
                    report.failed += 1
                    report.errors.append(f"{image_path.name}: {exc}")
                if progress_callback:
                    progress_callback(index, len(images))
            if report.failed == 0 and len(batch) == len(images):
                try:
                    self.annotation_service.save_coco_batch(batch, output_annotation_dir, presets)
                except Exception as exc:
                    report.errors.append(str(exc))
                    report.failed = len(batch)
                    report.succeeded = 0
            return report
        keypoint_names: list[str] = []
        pose_schema: tuple[str, ...] | None = None
        total = len(images)
        for index, image_path in enumerate(images, start=1):
            if cancel_callback and cancel_callback():
                break
            try:
                target = None if output_format == "coco" else output_annotation_dir / (
                    f"{image_path.stem}.xml" if output_format == "voc" else f"{image_path.stem}.txt"
                )
                if target is not None and target.exists() and not options.overwrite:
                    report.skipped += 1
                else:
                    result = self.annotation_service.load(image_path, annotation_dir, settings)
                    if result.error:
                        raise ValueError(result.error)
                    if output_task == "yolo_pose" and not keypoint_names:
                        keypoint_names = [item.name for item in next((annotation.keypoints for annotation in result.annotations if annotation.keypoints), [])]
                    if output_task == "yolo_pose":
                        for annotation in result.annotations:
                            schema = tuple(item.name for item in annotation.keypoints)
                            if pose_schema is None:
                                pose_schema = schema
                            elif schema != pose_schema:
                                raise ValueError("inconsistent keypoint schema for YOLO Pose output")
                    output_settings = ProjectSettings(
                        annotation_format=output_format,
                        dataset_task=output_task,
                        annotation_dir=output_annotation_dir,
                        label_presets=presets,
                    )
                    saved = self.annotation_service.save(image_path, result.annotations, output_annotation_dir, output_settings)
                    if not saved.ok:
                        raise OSError(saved.error or "conversion save failed")
                    if structured_output:
                        shutil.copy2(image_path, output_image_dir / image_path.name)
                    report.succeeded += 1
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{image_path.name}: {exc}")
            if progress_callback:
                progress_callback(index, total)
        if output_format == "yolo" and output_task == "yolo_pose":
            self._write_yolo_pose_yaml(options.output_path, presets, keypoint_names)
        return report

    @staticmethod
    def _canonical_format(value: str) -> str:
        normalized = str(value).strip().lower().replace(" ", "")
        aliases = {"yolo": "yolo", "voc": "voc", "pascalvoc": "voc", "coco": "coco"}
        if normalized not in aliases:
            raise ValueError(f"unsupported dataset format: {value}")
        return aliases[normalized]

    @staticmethod
    def _resolve_source(source: Path, format_name: str) -> tuple[Path, Path, bool]:
        source = Path(source)
        format_name = ConversionService._canonical_format(format_name)
        if source.is_dir():
            try:
                detected = DatasetDetector.detect(source)
                if detected.format_name == format_name:
                    return detected.image_dir, detected.annotation_dir, True
            except ValueError:
                pass
        candidates = (
            (source / "JPEGImages", source / "Annotations"),
            (source / "images", source / "Annotations"),
        ) if format_name == "voc" else (
            (source / "images", source / "annotations"),
            (source, source / "annotations"),
        ) if format_name == "coco" else (
            (source / "images", source / "labels"),
            (source / "images" / "train", source / "labels" / "train"),
            (source / "train" / "images", source / "train" / "labels"),
        )
        for image_dir, annotation_dir in candidates:
            if image_dir.is_dir() and annotation_dir.is_dir():
                return image_dir, annotation_dir, True
        if source.is_dir():
            sibling = source.parent / ("Annotations" if format_name == "voc" else "annotations" if format_name == "coco" else "labels")
            if sibling.is_dir():
                return source, sibling, True
        if source.is_dir():
            return source, source, False
        raise ValueError(f"源路径不存在: {source}")

    @staticmethod
    def _complete_presets(images: list[Path], annotation_dir: Path, format_name: str, presets: list, source_root: Path | None = None) -> list:
        result = list(presets)
        by_name = {preset.name for preset in result}
        by_id = {preset.class_id for preset in result}
        next_id = max(by_id, default=-1) + 1
        names: set[str] = set()
        if format_name.lower() == "voc":
            for image in images:
                xml_path = annotation_dir / f"{image.stem}.xml"
                if not xml_path.exists():
                    continue
                root = ET.parse(xml_path).getroot()
                names.update((node.findtext("name") or "").strip() for node in root.findall("object"))
        elif format_name.lower() == "yolo":
            classes_path = (source_root or annotation_dir) / "classes.txt"
            if classes_path.exists():
                class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines()]
                for class_id, name in enumerate(class_names):
                    if name and class_id not in by_id:
                        result.append(LabelPreset(name, class_id, QColor.fromHsv((class_id * 47) % 360, 210, 245).name()))
                        by_name.add(name); by_id.add(class_id)
            ids: set[int] = set()
            for image in images:
                txt_path = annotation_dir / f"{image.stem}.txt"
                if txt_path.exists():
                    for line in txt_path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            ids.add(int(line.split()[0]))
            names.update(f"class_{class_id}" for class_id in sorted(ids) if class_id not in by_id)
        else:
            import json
            json_files = sorted(annotation_dir.glob("*.json"))
            if json_files:
                document = json.loads(json_files[0].read_text(encoding="utf-8"))
                names.update(str(item.get("name", "")).strip() for item in document.get("categories", []))
        for name in sorted(name for name in names if name and name not in by_name):
            while next_id in by_id:
                next_id += 1
            result.append(LabelPreset(name, next_id, QColor.fromHsv((next_id * 47) % 360, 210, 245).name()))
            by_name.add(name); by_id.add(next_id); next_id += 1
        return result

    @staticmethod
    def _write_yolo_classes(root: Path, presets: list) -> None:
        by_id = {preset.class_id: preset.name for preset in presets}
        max_id = max(by_id, default=-1)
        (root / "classes.txt").write_text(
            "\n".join(by_id.get(index, f"class_{index}") for index in range(max_id + 1)) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_yolo_pose_yaml(root: Path, presets: list, keypoint_names: list[str]) -> None:
        names = "\n".join(f"  {preset.class_id}: {preset.name}" for preset in presets)
        keypoint_names = keypoint_names or list(COCO_PERSON_KEYPOINTS)
        keypoints = len(keypoint_names)
        content = "\n".join([
            f"path: {root.as_posix()}",
            "train: images",
            "val: images",
            "names:",
            names,
            f"kpt_shape: [{keypoints}, 3]",
            "kpt_names:",
            "  - [" + ", ".join(keypoint_names) + "]",
        ]) + "\n"
        (root / "data.yaml").write_text(content, encoding="utf-8")
