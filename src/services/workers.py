from __future__ import annotations

from pathlib import Path
import os

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor

from src.models.annotation import Annotation, LabelPreset
from src.models.project import ImageRecord, ProjectSettings
from .annotation_service import AnnotationService
from .onnx_service import YoloOnnxDetector
from .project_service import ProjectService


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
    modelReady = Signal(str, object)
    annotationReady = Signal(str, object)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, records: list[ImageRecord], settings: ProjectSettings) -> None:
        super().__init__()
        self.records = [ImageRecord(path=item.path, width=item.width, height=item.height,
                                    file_format=item.file_format, file_size=item.file_size,
                                    status=item.status, metadata_loaded=item.metadata_loaded)
                        for item in records]
        self.settings = ProjectSettings.from_dict(settings.to_dict())
        self.cancelled = False

    def run(self) -> None:
        try:
            if not self.settings.onnx_model_path:
                raise ValueError("请先在应用设置中选择 ONNX 模型")
            detector = YoloOnnxDetector()
            detector.load(self.settings.onnx_model_path)
            if detector.task == "pose":
                self.settings.dataset_task = "yolo_pose"
            self.modelReady.emit(detector.task, list(detector.keypoint_names))
            service = AnnotationService()
            total = len(self.records)
            size = (self.settings.input_width, self.settings.input_height)
            for current, record in enumerate(self.records, 1):
                if self.cancelled:
                    break
                from PIL import Image
                with Image.open(record.path) as image:
                    annotations = detector.predict(image, self.settings.label_presets, size,
                                                    self.settings.confidence_threshold,
                                                    self.settings.nms_threshold)
                if self.settings.auto_save and self.settings.annotation_dir is not None:
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
                classes_path = self.image_dir.parent / "classes.txt"
                if classes_path.exists():
                    classes_path.write_text("\n".join(preset.name for preset in self.settings.label_presets) + "\n", encoding="utf-8")
            self.finished.emit("" if result.ok else (result.error or "保存标注失败"))
        except Exception as exc:
            self.finished.emit(str(exc))
