from __future__ import annotations

from pathlib import Path
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor

from src.models.annotation import Annotation, LabelPreset
from src.models.project import ImageRecord, ProjectSettings
from .annotation_service import AnnotationService
from .onnx_service import YoloOnnxDetector
from .project_service import ProjectService
from .dataset_index import DatasetIndexRepository, IndexedImage
from .dataset_session import DatasetScanResult
from .yolo_metadata import yolo_class_names


class DatasetStatisticsWorker(QObject):
    progress = Signal(int, int, object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, records, image_dir: Path, annotation_dir: Path, settings: ProjectSettings, presets: list[LabelPreset]) -> None:
        super().__init__()
        self.records = [ImageRecord(path=r.path, width=r.width, height=r.height, file_format=r.file_format, file_size=r.file_size, status=r.status, metadata_loaded=r.metadata_loaded) for r in records] if records is not None else None
        self.image_dir, self.annotation_dir = Path(image_dir), Path(annotation_dir)
        self.settings = ProjectSettings.from_dict(settings.to_dict())
        self.settings.label_presets = list(presets or self.settings.label_presets)
        self._class_names = {p.class_id: p.name for p in self.settings.label_presets}
        self.cancelled = False

    def run(self) -> None:
        try:
            service = AnnotationService()
            repository = DatasetIndexRepository(self.image_dir.parent.resolve(), self.image_dir, self.annotation_dir, self.settings.annotation_format)
            if self.records is None:
                total = repository.count()
                pages = repository.iter_pages(500)
            else:
                total = len(self.records)
                pages = (self.records[index:index + 500] for index in range(0, total, 500))
            self.progress.emit(0, total, self._snapshot(total, 0, 0, Counter()))
            index = service.build_index(self.annotation_dir, self.settings.annotation_format, cancel_callback=lambda: self.cancelled) if self.settings.annotation_format == "coco" else None
            labeled = total_labels = 0
            labels: Counter[str] = Counter()
            workers = 1 if self.settings.annotation_format == "coco" else min(6, max(1, os.cpu_count() or 1), max(1, total))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="annotation-stats") as pool:
                processed = 0
                for page in pages:
                    if self.cancelled:
                        return
                    records = [
                        ImageRecord(
                            path=item.path, width=item.width, height=item.height,
                            file_format=item.path.suffix.lstrip(".").upper(),
                            file_size=item.file_size, annotations=[],
                            status=getattr(item, "annotation_status", "pending"),
                            metadata_loaded=False,
                        ) if isinstance(item, IndexedImage) else item
                        for item in page
                    ]
                    for start in range(0, len(records), 256):
                        futures = [pool.submit(self._read_labels, record, index) for record in records[start:start + 256]]
                        for future in futures:
                            if self.cancelled:
                                return
                            values = future.result()
                            if values:
                                labeled += 1; total_labels += len(values); labels.update(values)
                            processed += 1
                            self.progress.emit(processed, total, self._snapshot(total, labeled, total_labels, labels))
            self.finished.emit(self._snapshot(total, labeled, total_labels, labels))
        except Exception as exc:
            self.failed.emit(str(exc))

    @staticmethod
    def _snapshot(total, labeled, total_labels, labels):
        return {"total_images": total, "labeled_images": labeled, "percentage": labeled / total * 100.0 if total else 0.0, "total_labels": total_labels, "label_counts": dict(sorted(labels.items()))}

    def _read_labels(self, record, index):
        if self.settings.annotation_format == "coco":
            relative = record.path.relative_to(self.image_dir).as_posix()
            image = index.coco_images.get(relative)
            if image is None:
                matches = [item for name, item in index.coco_images.items() if "/" in name and Path(name).name == record.path.name]
                image = matches[0] if len(matches) == 1 else None
            if not image:
                return []
            return [str(index.coco_categories.get(item.get("category_id"), f"class_{item.get('category_id')}")) for item in index.coco_annotations.get(image.get("id"), []) if len(item.get("bbox", [])) >= 4]
        target = self.annotation_dir / record.path.parent.relative_to(self.image_dir) / f"{record.path.stem}{'.xml' if self.settings.annotation_format == 'voc' else '.txt'}"
        if not target.exists():
            target = self.annotation_dir / f"{record.path.stem}{'.xml' if self.settings.annotation_format == 'voc' else '.txt'}"
        if not target.exists():
            return []
        try:
            if self.settings.annotation_format == "voc":
                return [name for name in (item.findtext("name", "").strip() for item in ET.parse(target).getroot().findall("object")) if name]
            values = []
            for line in target.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts and parts[0].isdigit():
                    values.append(self._class_names.get(int(parts[0]), f"class_{parts[0]}"))
            return values
        except (OSError, ValueError, ET.ParseError):
            return []


class DatasetScanWorker(QObject):
    progress = Signal(int, int)
    partial = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, image_dir: Path, annotation_dir: Path, settings: ProjectSettings, dataset_root: Path | None = None, session_id: str = "") -> None:
        super().__init__()
        self.image_dir, self.annotation_dir = Path(image_dir), Path(annotation_dir)
        self.settings = ProjectSettings.from_dict(settings.to_dict())
        self.dataset_root, self.session_id = Path(dataset_root or image_dir.parent).resolve(), session_id
        self.cancelled = False

    def run(self) -> None:
        try:
            repository = DatasetIndexRepository(self.dataset_root, self.image_dir, self.annotation_dir, self.settings.annotation_format)
            presets = self._discover_presets(100)
            if repository.is_complete():
                self.partial.emit(DatasetScanResult([], presets, repository.count(), True, self.session_id))
            repository.set_complete(False)
            indexed = 0; first = False
            for batch in repository.scan_paths(lambda: self.cancelled, 500, [p.name for p in presets]):
                repository.upsert_batch(batch); indexed += len(batch)
                if not first:
                    self.partial.emit(DatasetScanResult([self._record(item) for item in batch], presets, 0, True, self.session_id)); first = True
                self.progress.emit(indexed, 0)
            if self.cancelled:
                return
            repository.prune_missing(lambda: self.cancelled); total = repository.count(); repository.set_complete(True)
            self.progress.emit(total, total); self.finished.emit(DatasetScanResult([], presets, total, True, self.session_id))
        except Exception as exc:
            self.failed.emit(str(exc))

    @staticmethod
    def _record(item: IndexedImage) -> ImageRecord:
        return ImageRecord(path=item.path, width=item.width, height=item.height, file_format=item.path.suffix.lstrip(".").upper(), file_size=item.file_size, annotations=[], status="pending", metadata_loaded=False)

    def _discover_presets(self, sample_limit=None):
        names = []; root = self.image_dir.parent
        if self.settings.annotation_format == "yolo":
            for path in (root / "classes.txt", root.parent / "classes.txt"):
                if path.exists():
                    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]; break
            if not names:
                names = [name for name in yolo_class_names(root) if name]
        elif self.settings.annotation_format == "voc":
            for index, path in enumerate(self.annotation_dir.rglob("*.xml")):
                if sample_limit is not None and index >= sample_limit: break
                try: names.extend((node.findtext("name") or "").strip() for node in ET.parse(path).getroot().findall("object"))
                except (OSError, ET.ParseError): pass
        else:
            import json
            for path in sorted(self.annotation_dir.glob("*.json")):
                try:
                    names.extend(str(item.get("name", "")).strip() for item in json.loads(path.read_text(encoding="utf-8")).get("categories", [])); break
                except (OSError, json.JSONDecodeError): pass
        names = list(dict.fromkeys(name for name in names if name))
        if not names: return list(self.settings.label_presets)
        colors = [p.color for p in self.settings.label_presets]
        from PySide6.QtGui import QColor
        return [LabelPreset(name, i, colors[i] if i < len(colors) else QColor.fromHsv((i * 47) % 360, 210, 245).name()) for i, name in enumerate(names)]


class DatasetCountWorker(QObject):
    finished = Signal(int, str)

    def __init__(self, image_dir: Path, session_id: str) -> None:
        super().__init__()
        self.image_dir, self.session_id, self.cancelled = Path(image_dir), session_id, False

    def run(self) -> None:
        total = 0
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for root, _dirs, files in os.walk(self.image_dir):
            if self.cancelled:
                return
            total += sum(1 for name in files if Path(name).suffix.lower() in extensions)
        self.finished.emit(total, self.session_id)


class SingleImageAnnotationWorker(QObject):
    finished = Signal(str, int, int, str, object, str)

    def __init__(self, record: ImageRecord, image_dir: Path, annotation_dir: Path, settings: ProjectSettings) -> None:
        super().__init__()
        self.record = ImageRecord(path=record.path, width=record.width, height=record.height,
                                  file_format=record.file_format, file_size=record.file_size,
                                  status=record.status, metadata_loaded=record.metadata_loaded)
        self.image_dir, self.annotation_dir = Path(image_dir), Path(annotation_dir)
        self.settings = ProjectSettings.from_dict(settings.to_dict())

    def run(self) -> None:
        try:
            from PIL import Image
            with Image.open(self.record.path) as image:
                width, height = image.size
                file_format = image.format or self.record.path.suffix.lstrip(".").upper()
            relative_parent = self.record.path.parent.relative_to(self.image_dir)
            target_dir = self.annotation_dir if self.settings.annotation_format == "coco" else self.annotation_dir / relative_parent
            if self.settings.annotation_format == "yolo":
                self.settings.label_presets = self._dataset_yolo_presets()
            result = AnnotationService().load(self.record.path, target_dir, self.settings, image_size=(width, height))
            self.finished.emit(str(self.record.path), width, height, file_format, result.annotations, result.error or "")
        except (OSError, ValueError) as exc:
            self.finished.emit(str(self.record.path), 0, 0, self.record.path.suffix.lstrip(".").upper(), [], str(exc))

    def _dataset_yolo_presets(self) -> list[LabelPreset]:
        root = self.image_dir.parent
        names: list[str] = []
        for path in (root / "classes.txt", root.parent / "classes.txt"):
            if path.exists():
                names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                break
        if not names:
            names = [preset.name for preset in self.settings.label_presets]
        colors = [preset.color for preset in self.settings.label_presets]
        return [LabelPreset(name, index, colors[index] if index < len(colors) else QColor.fromHsv((index * 47) % 360, 210, 245).name()) for index, name in enumerate(names)]


class AutoLabelWorker(QObject):
    progress = Signal(int, int)
    modelReady = Signal(str, object, object)
    annotationReady = Signal(str, object)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, records: list[ImageRecord] | None, settings: ProjectSettings) -> None:
        super().__init__()
        self.records = None if records is None else [
            ImageRecord(path=item.path, width=item.width, height=item.height,
                        file_format=item.file_format, file_size=item.file_size,
                        status=item.status, metadata_loaded=item.metadata_loaded)
            for item in records
        ]
        self.settings = ProjectSettings.from_dict(settings.to_dict())
        self.cancelled = False

    def run(self) -> None:
        try:
            if not self.settings.onnx_model_path:
                raise ValueError("请先在应用设置中选择 ONNX 模型")
            detector = YoloOnnxDetector()
            detector.load(self.settings.onnx_model_path)
            if detector.class_names and self.settings.annotation_format == "yolo":
                dataset_names = [
                    preset.name for preset in sorted(
                        self.settings.label_presets, key=lambda preset: preset.class_id,
                    )
                ]
                if dataset_names and dataset_names != detector.class_names:
                    raise ValueError(
                        "ONNX model classes do not match the opened YOLO dataset classes"
                    )
            if detector.task == "pose" and self.settings.annotation_format == "yolo":
                self.settings.dataset_task = "yolo_pose"
            elif detector.task in {"segment", "segmentation"} and self.settings.annotation_format == "yolo":
                self.settings.dataset_task = "yolo_segmentation"
            if detector.task in {"pose", "segment", "segmentation"} and self.settings.annotation_format == "voc":
                raise ValueError(
                    "Pascal VOC does not support ONNX pose or segmentation annotations"
                )
            model_presets = self._model_presets(detector.class_names)
            self.settings.label_presets = model_presets
            self.modelReady.emit(
                detector.task, list(detector.keypoint_names), list(detector.class_names),
            )
            service = AnnotationService()
            records, total = self._records_and_total()
            size = (self.settings.input_width, self.settings.input_height)
            for current, record in enumerate(records, 1):
                if self.cancelled:
                    break
                from PIL import Image
                with Image.open(record.path) as image:
                    annotations = detector.predict(image, model_presets, size,
                                                    self.settings.confidence_threshold,
                                                    self.settings.nms_threshold)
                if self.settings.annotation_dir is not None:
                    target_dir = self.settings.annotation_dir
                    if self.settings.annotation_format != "coco" and self.settings.image_dir is not None:
                        try:
                            target_dir /= record.path.parent.relative_to(self.settings.image_dir)
                        except ValueError:
                            pass
                    saved = service.save(record.path, annotations, target_dir, self.settings)
                    if not saved.ok:
                        raise OSError(saved.error or f"保存自动标注失败: {record.path.name}")
                self.annotationReady.emit(str(record.path), annotations)
                self.progress.emit(current, total)
            self.finished.emit()
        except Exception as exc:
            self.failed.emit(str(exc))

    def _records_and_total(self):
        if self.records is not None:
            return iter(self.records), len(self.records)
        if self.settings.image_dir is None or self.settings.annotation_dir is None:
            return iter(()), 0
        repository = DatasetIndexRepository(
            self.settings.image_dir.parent.resolve(),
            self.settings.image_dir,
            self.settings.annotation_dir,
            self.settings.annotation_format,
        )
        total = repository.count()

        def records():
            for page in repository.iter_pages(500):
                if self.cancelled:
                    return
                for item in page:
                    yield ImageRecord(
                        path=item.path, width=item.width, height=item.height,
                        file_format=item.path.suffix.lstrip(".").upper(),
                        file_size=item.file_size, status=item.annotation_status,
                        metadata_loaded=False,
                    )

        return records(), total

    def _model_presets(self, class_names: list[str]) -> list[LabelPreset]:
        if not class_names:
            return list(self.settings.label_presets)
        colors = {preset.class_id: preset.color for preset in self.settings.label_presets}
        return [
            LabelPreset(
                name, class_id,
                colors.get(class_id, QColor.fromHsv((class_id * 47) % 360, 210, 245).name()),
            )
            for class_id, name in enumerate(class_names)
        ]


class SaveWorker(QObject):
    finished = Signal(str)

    def __init__(self, project_path, image_path, annotations, settings) -> None:
        super().__init__()
        self.args = project_path, image_path, annotations, ProjectSettings.from_dict(settings.to_dict())
        self.args = self.args[0], self.args[1], [Annotation.from_dict(item.to_dict()) for item in annotations], self.args[3]

    def run(self) -> None:
        try:
            ProjectService(AnnotationService()).save_current(*self.args)
            self.finished.emit("")
        except Exception as exc:
            self.finished.emit(str(exc))


class DatasetAnnotationSaveWorker(QObject):
    finished = Signal(str)

    def __init__(self, image_path: Path, image_dir: Path, annotation_dir: Path,
                 annotations: list[Annotation], settings: ProjectSettings) -> None:
        super().__init__()
        self.image_path, self.image_dir, self.annotation_dir = Path(image_path), Path(image_dir), Path(annotation_dir)
        self.annotations = [Annotation.from_dict(item.to_dict()) for item in annotations]
        self.settings = ProjectSettings.from_dict(settings.to_dict())

    def run(self) -> None:
        try:
            if self.settings.annotation_format == "yolo":
                known = {preset.name for preset in self.settings.label_presets}
                next_id = max((preset.class_id for preset in self.settings.label_presets), default=-1) + 1
                for annotation in self.annotations:
                    if annotation.label not in known:
                        self.settings.label_presets.append(LabelPreset(annotation.label, next_id, annotation.color))
                        known.add(annotation.label)
                        next_id += 1
            relative_parent = self.image_path.parent.relative_to(self.image_dir)
            target_dir = self.annotation_dir if self.settings.annotation_format == "coco" else self.annotation_dir / relative_parent
            result = AnnotationService().save(self.image_path, self.annotations, target_dir, self.settings)
            if self.settings.annotation_format == "yolo":
                # classes.txt is the authoritative class-id mapping for YOLO
                # label files; bootstrap it for datasets that have none yet.
                classes_path = self.image_dir.parent / "classes.txt"
                classes_path.write_text("\n".join(preset.name for preset in self.settings.label_presets) + "\n", encoding="utf-8")
            self.finished.emit("" if result.ok else (result.error or "保存标注失败"))
        except Exception as exc:
            self.finished.emit(str(exc))
