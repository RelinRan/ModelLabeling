from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET
from typing import Callable

from PySide6.QtGui import QColor
import yaml

from src.models.annotation import Annotation, LabelPreset, ShapeType
from PySide6.QtCore import QPointF
from src.utils.geometry import rect_from_points
from src.models.project import ProjectSettings
from .annotation_service import AnnotationService
from .dataset_detector import DatasetDetector
from .yolo_metadata import yolo_class_names, yolo_keypoint_names, yolo_keypoint_shape, load_yolo_metadata


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
    force_structured_output: bool = False
    plain_image_source: bool = False
    flatten_output: bool = False
    merge_polygon_parts: bool = False


@dataclass
class ConversionReport:
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] | None = None
    cancelled: bool = False

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []




def _flatten_relative(relative: Path) -> Path:
    """Flatten a nested source path while retaining enough context for uniqueness."""
    if len(relative.parts) <= 1:
        return relative
    stem = "__".join(relative.with_suffix("").parts)
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in stem)
    return Path(safe + relative.suffix)

class ConversionService:
    def __init__(self, annotation_service: AnnotationService | None = None) -> None:
        self.annotation_service = annotation_service or AnnotationService()

    def convert(
        self,
        options: ConversionOptions,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> ConversionReport:
        source_root = Path(options.source_path).resolve()
        output_root = Path(options.output_path).resolve()
        if (source_root == output_root or source_root in output_root.parents
                or output_root in source_root.parents):
            raise ValueError("Source and output directories must not overlap")
        if output_root.exists() and not output_root.is_dir():
            raise ValueError("Output path must be a directory")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.staging-",
                                         dir=output_root.parent) as temporary:
            stage_root = Path(temporary) / "dataset"
            stage_root.mkdir()
            return self._convert(options, stage_root, progress_callback, cancel_callback)

    def _convert(
        self, options: ConversionOptions, stage_root: Path,
        progress_callback: Callable[[int, int], None] | None,
        cancel_callback: Callable[[], bool] | None,
    ) -> ConversionReport:
        report = ConversionReport()
        source_root = Path(options.source_path).resolve()
        output_root = Path(options.output_path).resolve()
        source_format = self._canonical_format(options.source_format)
        output_format = self._canonical_format(options.output_format)
        if output_format == "coco" and output_root.exists() and not options.overwrite and any(output_root.iterdir()):
            raise ValueError("COCO output directory is not empty; enable overwrite or choose an empty output")
        source_task = options.source_task
        if source_task is None and source_format == "yolo" and not options.plain_image_source:
            source_task = DatasetDetector.detect(options.source_path).task_name
        source_task = source_task or (source_format if source_format != "yolo" else "yolo_detection")
        output_task = options.output_task or (
            source_task if output_format == source_format else
            output_format if output_format != "yolo" else "yolo_detection"
        )
        pose_shape = yolo_keypoint_shape(source_root) if source_task == "yolo_pose" else None
        if options.plain_image_source:
            image_dir, annotation_dir, structured = options.source_path, options.source_path, False
        else:
            image_dir, annotation_dir, structured = self._resolve_source(options.source_path, source_format)
        images = sorted(
            item for item in image_dir.rglob("*")
            if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not images:
            raise ValueError("Source dataset contains no supported images")
        if progress_callback:
            progress_callback(0, len(images))
        if cancel_callback and cancel_callback():
            report.cancelled = True
            return report
        output_names: set[str] = set()
        annotation_names: set[str] = set()
        for image in images:
            relative = image.relative_to(image_dir)
            relative = _flatten_relative(relative) if options.flatten_output else relative
            image_name = relative.as_posix().casefold()
            annotation_name = relative.with_suffix(".txt").as_posix().casefold()
            if image_name in output_names or (output_format != "coco" and annotation_name in annotation_names):
                raise ValueError(f"Multiple source images map to the same output file: {relative}")
            output_names.add(image_name)
            annotation_names.add(annotation_name)
        presets = self._complete_presets(
            images, annotation_dir, source_format, options.presets, options.source_path, image_dir,
        )
        if output_format == "yolo" and output_root.exists() and not options.overwrite:
            try:
                existing = DatasetDetector.detect(output_root, allow_plain_images=False)
            except ValueError:
                existing = None
            if existing and existing.format_name == "yolo":
                existing_names = yolo_class_names(output_root)
                classes = output_root / "classes.txt"
                if classes.is_file():
                    existing_names = classes.read_text(encoding="utf-8").splitlines()
                by_id = {preset.class_id: preset.name for preset in presets}
                output_names = [by_id.get(i, f"class_{i}") for i in range(max(by_id, default=-1) + 1)]
                if existing_names and existing_names != output_names:
                    raise ValueError("Existing YOLO output classes differ; choose an empty output or enable overwrite")
                if existing.task_name != output_task:
                    raise ValueError("Existing YOLO output task differs; choose an empty output or enable overwrite")
                if output_task == "yolo_pose":
                    existing_shape = yolo_keypoint_shape(output_root)
                    if existing_shape and pose_shape and existing_shape != pose_shape:
                        raise ValueError("Existing YOLO output kpt_shape differs")
                    existing_points = yolo_keypoint_names(output_root)
                    source_points = yolo_keypoint_names(source_root) if source_task == "yolo_pose" else []
                    if existing_points and source_points and existing_points != source_points:
                        raise ValueError("Existing YOLO output kpt_names differ")
        structured_output = (
            structured
            or output_format == "coco"
            or options.force_structured_output
            or output_format == "yolo"
        )
        if structured_output:
            if output_format == "yolo":
                real_output_image_dir = output_root / "images"
                real_output_annotation_dir = output_root / "labels"
            elif output_format == "coco":
                real_output_image_dir = output_root / "images"
                real_output_annotation_dir = output_root / "annotations"
            else:
                real_output_image_dir = output_root / "JPEGImages"
                real_output_annotation_dir = output_root / "Annotations"
        else:
            real_output_image_dir = output_root
            real_output_annotation_dir = output_root
        output_image_dir = stage_root / real_output_image_dir.relative_to(output_root)
        output_annotation_dir = stage_root / real_output_annotation_dir.relative_to(output_root)
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_annotation_dir.mkdir(parents=True, exist_ok=True)
        publication = (options.overwrite, output_format,
                       output_image_dir.relative_to(stage_root), output_annotation_dir.relative_to(stage_root))
        if output_format == "yolo":
            self._write_yolo_classes(stage_root, presets)
        settings = ProjectSettings(
            image_dir=image_dir,
            annotation_format=source_format,
            dataset_task=source_task,
            annotation_dir=annotation_dir,
            label_presets=presets,
        )
        if output_format == "coco":
            for index, image_path in enumerate(images, start=1):
                if cancel_callback and cancel_callback():
                    report.cancelled = True
                    break
                try:
                    result = self.annotation_service.load(image_path, annotation_dir, settings)
                    if result.error:
                        raise ValueError(result.error)
                    result.annotations = self._adapt_geometry(result.annotations, output_task, options.merge_polygon_parts)
                    relative = image_path.relative_to(image_dir)
                    output_relative = _flatten_relative(relative) if options.flatten_output else relative
                    self.annotation_service.save_coco_record(
                        image_path, result.annotations, output_annotation_dir,
                        presets, output_relative.as_posix(),
                    )
                    output_image_path = output_image_dir / output_relative
                    output_image_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_path, output_image_path)
                    report.succeeded += 1
                except Exception as exc:
                    report.failed += 1
                    report.errors.append(f"{image_path.name}: {exc}")
                if progress_callback:
                    progress_callback(index, len(images))
            if report.failed == 0 and not report.cancelled and report.succeeded == len(images):
                try:
                    AnnotationService.export_coco(output_annotation_dir)
                except Exception as exc:
                    report.errors.append(str(exc))
                    report.failed = report.succeeded
                    report.succeeded = 0
            if cancel_callback and cancel_callback():
                report.cancelled = True
            return self._finalize(report, stage_root, output_root, *publication)
        keypoint_names = yolo_keypoint_names(source_root) if source_task == "yolo_pose" else []
        pose_schema: tuple[str, ...] | None = None
        total = len(images)
        for index, image_path in enumerate(images, start=1):
            if cancel_callback and cancel_callback():
                report.cancelled = True
                break
            try:
                relative = image_path.relative_to(image_dir)
                output_relative = _flatten_relative(relative) if options.flatten_output else relative
                target = None if output_format == "coco" else output_annotation_dir / output_relative.with_suffix(
                    ".xml" if output_format == "voc" else ".txt"
                )
                real_target = None if target is None else real_output_annotation_dir / output_relative.with_suffix(
                    ".xml" if output_format == "voc" else ".txt"
                )
                if real_target is not None and real_target.exists() and not options.overwrite:
                    report.skipped += 1
                else:
                    result = self.annotation_service.load(image_path, annotation_dir, settings)
                    if result.error:
                        raise ValueError(result.error)
                    result.annotations = self._adapt_geometry(result.annotations, output_task, options.merge_polygon_parts)
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
                        image_dir=(image_path.parent if options.flatten_output else image_dir),
                        annotation_format=output_format,
                        dataset_task=output_task,
                        annotation_dir=output_annotation_dir,
                        label_presets=presets,
                        keypoint_dimensions=pose_shape[1] if pose_shape else 3,
                    )
                    saved = self.annotation_service.save(image_path, result.annotations, output_annotation_dir, output_settings)
                    if not saved.ok:
                        raise OSError(saved.error or "conversion save failed")
                    if options.flatten_output:
                        generated = output_annotation_dir / image_path.with_suffix(
                            ".xml" if output_format == "voc" else ".txt"
                        ).name
                        desired = output_annotation_dir / output_relative.with_suffix(
                            ".xml" if output_format == "voc" else ".txt"
                        )
                        if generated != desired and generated.exists():
                            desired.parent.mkdir(parents=True, exist_ok=True)
                            generated.replace(desired)
                    if structured_output:
                        output_image_path = output_image_dir / output_relative
                        output_image_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(image_path, output_image_path)
                    report.succeeded += 1
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{image_path.name}: {exc}")
            if progress_callback:
                progress_callback(index, total)
        if (
            output_format == "yolo"
            and output_task == "yolo_pose"
            and report.failed == 0
            and report.succeeded + report.skipped == total
        ):
            self._write_yolo_pose_yaml(stage_root, presets, keypoint_names,
                                       pose_shape, load_yolo_metadata(source_root) if source_format == "yolo" else {})
        elif (
            output_format == "yolo"
            and output_task == "yolo_segmentation"
            and report.failed == 0
            and report.succeeded + report.skipped == total
        ):
            self._write_yolo_segmentation_yaml(stage_root, presets)
        elif (output_format == "yolo" and report.failed == 0
              and report.succeeded + report.skipped == total):
            self._write_yolo_yaml(stage_root, presets, output_task)
        if cancel_callback and cancel_callback():
            report.cancelled = True
        return self._finalize(report, stage_root, output_root, *publication)

    @staticmethod
    def _finalize(report: ConversionReport, stage_root: Path, output_root: Path, overwrite: bool,
                  output_format: str, image_relative: Path, annotation_relative: Path) -> ConversionReport:
        """Publish only a complete conversion; never expose a partial target."""
        if report.cancelled or report.failed:
            return report
        backup_root = output_root.with_name(f".{output_root.name}.backup-{uuid.uuid4().hex}")
        moved_original = False
        published = False
        try:
            metadata_path = stage_root / "data.yaml"
            if metadata_path.is_file():
                document = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
                document["path"] = output_root.as_posix()
                AnnotationService._atomic_write_text(metadata_path, yaml.safe_dump(
                    document, allow_unicode=True, sort_keys=False, default_flow_style=None))
            if output_root.exists():
                output_root.replace(backup_root)
                moved_original = True
                annotation_suffix = {"yolo": ".txt", "voc": ".xml", "coco": ".json"}[output_format]
                for source in backup_root.rglob("*"):
                    if source.is_dir():
                        continue
                    relative = source.relative_to(backup_root)
                    if overwrite and (
                        relative.is_relative_to(annotation_relative) and source.suffix.lower() == annotation_suffix
                        or relative.is_relative_to(image_relative) and source.suffix.lower() in DatasetDetector.IMAGE_EXTENSIONS
                    ):
                        continue
                    if output_format == "coco" and source.name.startswith(".model_labeling.sqlite3"):
                        continue
                    target = stage_root / relative
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
            stage_root.replace(output_root)
            published = True
            if moved_original:
                shutil.rmtree(backup_root, ignore_errors=True)
        except Exception as exc:
            if moved_original and backup_root.exists():
                if output_root.exists():
                    shutil.rmtree(output_root, ignore_errors=True)
                backup_root.replace(output_root)
            elif published:
                shutil.rmtree(output_root, ignore_errors=True)
            shutil.rmtree(stage_root, ignore_errors=True)
            report.failed = max(1, report.failed)
            report.errors.append(f"publish conversion output failed: {exc}")
            report.succeeded = 0
        return report

    @staticmethod
    def _adapt_geometry(annotations, output_task, merge_polygon_parts=False):
        """Only equivalent representations; never invent keypoints or discard geometry."""
        result = []
        for original in annotations:
            annotation = Annotation.from_dict(original.to_dict())
            if (merge_polygon_parts and output_task == "yolo_segmentation"
                    and len(annotation.polygon_parts) > 1):
                from .segmentation_geometry import bridge_contours
                annotation.points = bridge_contours(annotation.polygon_parts)
                annotation.polygon_parts = []
            if annotation.shape_type == ShapeType.OBB and output_task in {"yolo_segmentation", "coco"}:
                annotation.shape_type = ShapeType.POLYGON
            elif annotation.shape_type in {ShapeType.RECTANGLE, ShapeType.SQUARE} and output_task == "yolo_obb":
                rect = rect_from_points(annotation.points)
                annotation.shape_type = ShapeType.OBB
                annotation.points = [rect.topLeft(), QPointF(rect.right(), rect.top()),
                                     rect.bottomRight(), QPointF(rect.left(), rect.bottom())]
            result.append(annotation)
        return result

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
    def _complete_presets(images: list[Path], annotation_dir: Path, format_name: str, presets: list, source_root: Path | None = None, image_dir: Path | None = None) -> list:
        result = list(presets)
        by_name = {preset.name for preset in result}
        by_id = {preset.class_id for preset in result}
        next_id = max(by_id, default=-1) + 1
        names: set[str] = set()
        if format_name.lower() == "voc":
            for image in images:
                directory = AnnotationService._annotation_directory_for_image(
                    image, annotation_dir, image_dir, ".xml",
                )
                xml_path = directory / f"{image.stem}.xml"
                if not xml_path.exists():
                    continue
                root = ET.parse(xml_path).getroot()
                names.update((node.findtext("name") or "").strip() for node in root.findall("object"))
        elif format_name.lower() == "yolo":
            classes_path = (source_root or annotation_dir) / "classes.txt"
            class_names: list[str] = []
            if classes_path.exists():
                class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines()]
            if not class_names:
                class_names = yolo_class_names(source_root or annotation_dir)
            if class_names:
                source_ids = set(range(len(class_names)))
                result = [preset for preset in result if preset.class_id not in source_ids]
                by_name = {preset.name for preset in result}
                by_id = {preset.class_id for preset in result}
                for class_id, name in enumerate(class_names):
                    if name:
                        result.append(LabelPreset(name, class_id, QColor.fromHsv((class_id * 47) % 360, 210, 245).name()))
                        by_name.add(name); by_id.add(class_id)
            ids: set[int] = set()
            for image in images:
                directory = AnnotationService._annotation_directory_for_image(
                    image, annotation_dir, image_dir, ".txt",
                )
                txt_path = directory / f"{image.stem}.txt"
                if txt_path.exists():
                    for line in txt_path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            try:
                                ids.add(int(line.split()[0]))
                            except (ValueError, IndexError):
                                # Let the normal annotation loader report the
                                # malformed file for its image; discovery must
                                # not abort the whole transaction first.
                                continue
            for class_id in sorted(ids - by_id):
                name = f"class_{class_id}"
                result.append(LabelPreset(name, class_id, QColor.fromHsv((class_id * 47) % 360, 210, 245).name()))
                by_id.add(class_id)
                by_name.add(name)
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
    def _write_yolo_pose_yaml(root: Path, presets: list, keypoint_names: list[str],
                              shape=None, source_metadata=None) -> None:
        if not keypoint_names and shape is None:
            # Empty images cannot establish a keypoint schema.
            raise ValueError("YOLO Pose output requires a keypoint schema")
        extra = {"kpt_shape": list(shape or (len(keypoint_names), 3))}
        if keypoint_names:
            extra["kpt_names"] = keypoint_names
        for key in ("flip_idx",):
            if key in (source_metadata or {}):
                extra[key] = source_metadata[key]
        ConversionService._write_yolo_yaml(root, presets, "yolo_pose", extra)

    @staticmethod
    def _write_yolo_yaml(root: Path, presets: list, task: str, extra=None, preserve_paths=False) -> None:
        by_id = {int(p.class_id): str(p.name) for p in presets}
        document = dict(load_yolo_metadata(root)) if (Path(root) / "data.yaml").exists() else {}
        for key in ("kpt_shape", "kpt_names", "flip_idx"):
            document.pop(key, None)
        document.update({
            "path": Path(root).resolve().as_posix(), "train": "images", "val": "images",
            "task": {"yolo_detection": "detect", "yolo_segmentation": "segment",
                     "yolo_pose": "pose", "yolo_obb": "obb"}[task],
            "names": {i: by_id.get(i, f"class_{i}") for i in range(max(by_id, default=-1) + 1)},
        })
        if preserve_paths and (Path(root) / "data.yaml").exists():
            original = load_yolo_metadata(root)
            for key in ("path", "train", "val", "test"):
                if key in original:
                    document[key] = original[key]
        document.update(extra or {})
        AnnotationService._atomic_write_text(Path(root) / "data.yaml",
            yaml.safe_dump(document, allow_unicode=True, sort_keys=False, default_flow_style=None))

    @staticmethod
    def _write_yolo_segmentation_yaml(root: Path, presets: list) -> None:
        """Write standard dataset paths and explicit task metadata for YOLO-Seg.

        The task declaration is essential for four-vertex polygons: their rows
        have the same number of values as YOLO OBB labels, so geometry alone
        cannot distinguish the two formats when the dataset is reopened.
        """
        ConversionService._write_yolo_yaml(root, presets, "yolo_segmentation")
