from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QFileDialog, QComboBox, QDialog, QFormLayout, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QWidget

from src.models.annotation import LabelPreset
from src.services.conversion_service import ConversionOptions, ConversionService
from src.services.dataset_detector import DatasetDetector
from .common_dialogs import AppDialog
from .form_layout import BUTTON_TOP_SPACING, configure_buttons, configure_form, set_confirm_button, set_content_margins, size_buttons


class ConversionWorker(QObject):
    progress = Signal(int, int)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, options: ConversionOptions) -> None:
        super().__init__()
        self.options = options
        self.cancelled = False

    def run(self) -> None:
        try:
            report = ConversionService().convert(
                self.options,
                progress_callback=lambda current, total: self.progress.emit(current, total),
                cancel_callback=lambda: self.cancelled,
            )
            self.completed.emit(report)
        except Exception as exc:
            self.failed.emit(str(exc))


class ConversionDialog(QDialog):
    def __init__(self, presets: list[LabelPreset], parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("conversionDialog")
        self.resize(460, 330)
        self.setFixedWidth(460)
        self.setWindowTitle("\u6570\u636e\u96c6\u8f6c\u6362")
        self.presets = list(presets)
        self.options: ConversionOptions | None = None
        self.source_path = QLineEdit(); self.output_path = QLineEdit()
        self.source_format = QComboBox(); self.source_format.addItem("COCO", "coco"); self.source_format.addItem("YOLO", "yolo"); self.source_format.addItem("Pascal VOC", "voc")
        self.output_format = QComboBox(); self.output_format.addItem("COCO", "coco"); self.output_format.addItem("YOLO", "yolo"); self.output_format.addItem("Pascal VOC", "voc")
        self.source_task = QComboBox(); self.output_task = QComboBox()
        self.source_format.currentIndexChanged.connect(self._refresh_task_options)
        self.output_format.currentIndexChanged.connect(self._refresh_task_options)
        self._refresh_task_options()
        form = configure_form(QFormLayout())
        form.addRow("\u6e90\u8def\u5f84", self._path_row(self.source_path, False)); form.addRow("\u6e90\u683c\u5f0f", self.source_format); form.addRow("\u6e90\u4efb\u52a1", self.source_task)
        form.addRow("\u8f6c\u8def\u5f84", self._path_row(self.output_path, True)); form.addRow("\u8f6c\u683c\u5f0f", self.output_format); form.addRow("\u8f6c\u4efb\u52a1", self.output_task)
        buttons = configure_buttons(QHBoxLayout()); buttons.addStretch(); self.cancel_button = QPushButton("\u53d6\u6d88"); self.confirm_button = QPushButton("\u786e\u8ba4")
        self.cancel_button.clicked.connect(self.reject); self.confirm_button.clicked.connect(self.accept)
        buttons.addWidget(self.cancel_button); buttons.addWidget(self.confirm_button)
        size_buttons(self.cancel_button, self.confirm_button)
        set_confirm_button(self.confirm_button)
        layout = QVBoxLayout(self); set_content_margins(layout); layout.setSpacing(0); layout.addLayout(form); layout.addSpacing(BUTTON_TOP_SPACING); layout.addLayout(buttons)
        self._load_last_conversion()

    def _load_last_conversion(self) -> None:
        settings = QSettings("RelinRan", "ModelLabeling")
        self.source_path.setText(str(settings.value("conversion/source_path", "") or ""))
        self.output_path.setText(str(settings.value("conversion/output_path", "") or ""))
        source_format = str(settings.value("conversion/source_format", "") or "")
        output_format = str(settings.value("conversion/output_format", "") or "")
        if source_format and self.source_format.findData(source_format) >= 0:
            self.source_format.setCurrentIndex(self.source_format.findData(source_format))
        if output_format and self.output_format.findData(output_format) >= 0:
            self.output_format.setCurrentIndex(self.output_format.findData(output_format))
        if self.source_path.text():
            source = Path(self.source_path.text())
            if source.is_dir():
                self._detect_source_format(source, show_error=False)

    def _refresh_task_options(self) -> None:
        def fill(combo: QComboBox, format_name: str) -> None:
            current = combo.currentData()
            options = {
                "yolo": [("YOLO Detection", "yolo_detection"), ("YOLO Segmentation", "yolo_segmentation"), ("YOLO Pose", "yolo_pose")],
                "coco": [("COCO", "coco")],
                "voc": [("Pascal VOC", "voc")],
            }.get(str(format_name), [])
            combo.blockSignals(True); combo.clear()
            for label, value in options: combo.addItem(label, value)
            index = combo.findData(current); combo.setCurrentIndex(index if index >= 0 else 0); combo.blockSignals(False)
        fill(self.source_task, self.source_format.currentData())
        fill(self.output_task, self.output_format.currentData())

    def _save_last_conversion(self, source: Path) -> None:
        settings = QSettings("RelinRan", "ModelLabeling")
        settings.setValue("conversion/source_path", str(source))
        settings.setValue("conversion/output_path", self.output_path.text().strip())
        settings.setValue("conversion/source_format", self.source_format.currentData())
        settings.setValue("conversion/output_format", self.output_format.currentData())
        settings.sync()

    def _path_row(self, editor: QLineEdit, output: bool) -> QWidget:
        row = QWidget(); layout = QHBoxLayout(row); layout.setContentsMargins(0, 0, 0, 0); layout.addWidget(editor)
        browse = QPushButton("\u6d4f\u89c8"); browse.clicked.connect(lambda: self._choose_path(editor, output)); layout.addWidget(browse); return row

    def _choose_path(self, editor: QLineEdit, output: bool) -> None:
        path = QFileDialog.getExistingDirectory(self, "\u9009\u62e9\u8f93\u51fa\u6587\u4ef6\u5939" if output else "\u9009\u62e9\u6e90\u6587\u4ef6\u5939", editor.text())
        if path:
            editor.setText(path)
            if not output and not self._detect_source_format(Path(path)):
                editor.clear()

    def _detect_source_format(self, source: Path, show_error: bool = True) -> bool:
        """Detect a supported dataset layout and select its source format."""
        if not source.is_dir():
            if show_error:
                AppDialog.information("\u6570\u636e\u96c6\u8f6c\u6362", "\u6e90\u8def\u5f84\u4e0d\u5b58\u5728\u3002", self)
            return False
        try:
            detected_info = DatasetDetector.detect(source)
            detected = detected_info.format_name
            self.source_format.setCurrentIndex(self.source_format.findData(detected))
            if detected_info.task_name and self.source_task.findData(detected_info.task_name) >= 0:
                self.source_task.setCurrentIndex(self.source_task.findData(detected_info.task_name))
            return True
        except ValueError:
            pass
        voc = (
            (source / "JPEGImages", source / "Annotations"),
            (source / "images", source / "Annotations"),
        )
        yolo = (
            (source / "images", source / "labels"),
            (source / "images" / "train", source / "labels" / "train"),
            (source / "train" / "images", source / "train" / "labels"),
        )
        # Check directory-based formats before COCO JSON because some YOLO
        # projects also keep a small annotations.json metadata file.
        detected = None
        if any(image_dir.is_dir() and annotation_dir.is_dir() for image_dir, annotation_dir in voc):
            detected = "voc"
        if detected is None and ((source / "classes.txt").is_file() or any(image_dir.is_dir() and annotation_dir.is_dir() for image_dir, annotation_dir in yolo)):
            detected = "yolo"
        is_coco = any((source / name).is_file() for name in ("annotations.json", "instances.json")) or bool(list(source.glob("instances_*.json")))
        if detected is None and is_coco:
            detected = "coco"
        if detected is None:
            if show_error:
                AppDialog.information("\u6570\u636e\u96c6\u8f6c\u6362", "\u4e0d\u652f\u6301\u6216\u65e0\u6cd5\u8bc6\u522b\u8be5\u6570\u636e\u96c6\u683c\u5f0f\u3002\u8bf7\u9009\u62e9 COCO\u3001YOLO \u6216 Pascal VOC \u6570\u636e\u96c6\u3002", self)
            return False
        self.source_format.setCurrentIndex(self.source_format.findData(detected))
        if detected == "yolo" and self.source_task.findData("yolo_detection") >= 0:
            self.source_task.setCurrentIndex(self.source_task.findData("yolo_detection"))
        return True

    def accept(self) -> None:
        source = Path(self.source_path.text())
        if not source.exists():
            AppDialog.information("\u8f6c\u6362\u5931\u8d25", "\u6e90\u8def\u5f84\u4e0d\u5b58\u5728", self)
            return
        if source.is_dir() and any(source.iterdir()) and not self._detect_source_format(source):
            return
        self.options = ConversionOptions(str(self.source_format.currentData()), source, str(self.output_format.currentData()), Path(self.output_path.text()), self.presets, source_task=str(self.source_task.currentData()), output_task=str(self.output_task.currentData()))
        self._save_last_conversion(source)
        super().accept()
