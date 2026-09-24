from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QLocale, QObject, QSettings, Signal
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from .common_dialogs import AppDialog
from .form_layout import (
    configure_buttons, configure_form, set_confirm_button,
    set_content_margins, size_buttons,
)


class _DirectoryDialog(QDialog):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(470)
        self.setWindowTitle(title)
        self.options = None

    def _last_path(self, operation: str, key: str) -> str:
        settings = QSettings("RelinRan", "ModelLabeling")
        return str(settings.value(f"video_tools/{operation}/{key}", "") or "")

    def _remember_path(self, operation: str, key: str, value: str) -> None:
        settings = QSettings("RelinRan", "ModelLabeling")
        settings.setValue(f"video_tools/{operation}/{key}", value)
        settings.sync()

    def _load_last_paths(self, operation: str, source: QLineEdit, output: QLineEdit) -> None:
        source.setText(self._last_path(operation, "source_dir"))
        output.setText(self._last_path(operation, "output_dir"))

    def _save_last_paths(self, operation: str, source: str, output: str) -> None:
        self._remember_path(operation, "source_dir", source)
        self._remember_path(operation, "output_dir", output)

    def _browse_row(self, editor: QLineEdit, chooser) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(editor)
        browse = QPushButton("\u6d4f\u89c8")
        browse.clicked.connect(lambda: chooser(editor))
        layout.addWidget(browse)
        return row

    def _path_row(self, editor: QLineEdit, output: bool) -> QWidget:
        return self._browse_row(editor, lambda target: self._choose_path(target, output))

    def _model_row(self, editor: QLineEdit) -> QWidget:
        return self._browse_row(editor, self._choose_model_file)

    def _choose_path(self, editor: QLineEdit, output: bool) -> None:
        title = "\u9009\u62e9\u8f93\u51fa\u6587\u4ef6\u5939" if output else "\u9009\u62e9\u6e90\u6587\u4ef6\u5939"
        path = QFileDialog.getExistingDirectory(self, title, editor.text())
        if path:
            editor.setText(path)

    def _choose_model_file(self, editor: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "\u9009\u62e9 YOLO \u6a21\u578b", editor.text(), "ONNX Model (*.onnx)"
        )
        if path:
            editor.setText(path)

    @staticmethod
    def _border_card(parent_layout: QVBoxLayout) -> QVBoxLayout:
        frame = QFrame()
        frame.setObjectName("sectionCard")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(12, 9, 12, 11)
        outer.setSpacing(7)
        content = QVBoxLayout()
        content.setSpacing(6)
        outer.addLayout(content)
        parent_layout.addWidget(frame)
        return content

    def _buttons(self, layout: QVBoxLayout) -> None:
        buttons = configure_buttons(QHBoxLayout())
        buttons.addStretch()
        cancel = QPushButton("\u53d6\u6d88")
        confirm = QPushButton("\u786e\u8ba4")
        cancel.clicked.connect(self.reject)
        confirm.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)
        size_buttons(cancel, confirm)
        set_confirm_button(confirm)
        layout.addStretch(1)
        layout.addLayout(buttons)


class VideoFrameDialog(_DirectoryDialog):
    def __init__(self, parent=None) -> None:
        super().__init__("\u89c6\u9891\u63d0\u5e27", parent)
        self.video_dir = QLineEdit()
        self.output_dir = QLineEdit()
        self.fps = QLineEdit("5")
        self.fps.setValidator(QIntValidator(1, 120, self.fps))
        layout = QVBoxLayout(self)
        set_content_margins(layout)
        layout.setSpacing(10)
        card = self._border_card(layout)
        form = configure_form(QFormLayout())
        form.addRow("\u89c6\u9891\u76ee\u5f55", self._path_row(self.video_dir, False))
        form.addRow("\u63d0\u5e27\u76ee\u5f55", self._path_row(self.output_dir, True))
        form.addRow("\u63d0\u5e27\u9891\u7387", self.fps)
        card.addLayout(form)
        self._load_last_paths("frames", self.video_dir, self.output_dir)
        self._buttons(layout)

    def accept(self) -> None:
        source = Path(self.video_dir.text().strip())
        output_text = self.output_dir.text().strip()
        if not source.is_dir() or not output_text or not self.fps.hasAcceptableInput():
            AppDialog.information(
                "\u63d0\u793a",
                "\u8bf7\u586b\u5199\u6709\u6548\u7684\u89c6\u9891\u76ee\u5f55\u3001\u63d0\u5e27\u76ee\u5f55\u548c1\u5230120\u4e4b\u95f4\u7684\u63d0\u5e27\u9891\u7387\u3002",
                self,
            )
            return
        self._save_last_paths("frames", str(source), output_text)
        self.options = {
            "source_dir": source,
            "output_dir": Path(output_text),
            "target_fps": int(self.fps.text()),
        }
        super().accept()


class DatasetSynthesisDialog(_DirectoryDialog):
    def __init__(self, parent=None) -> None:
        super().__init__("\u6570\u636e\u5408\u6210", parent)
        self.source_dir = QLineEdit()
        self.output_dir = QLineEdit()
        self.format = QComboBox()
        for label, value in (("COCO", "coco"), ("YOLO", "yolo"), ("Pascal VOC", "voc")):
            self.format.addItem(label, value)
        layout = QVBoxLayout(self)
        set_content_margins(layout)
        layout.setSpacing(10)
        card = self._border_card(layout)
        form = configure_form(QFormLayout())
        form.addRow("\u5e27\u56fe\u76ee\u5f55", self._path_row(self.source_dir, False))
        form.addRow("\u5408\u6210\u76ee\u5f55", self._path_row(self.output_dir, True))
        form.addRow("\u6570\u636e\u683c\u5f0f", self.format)
        card.addLayout(form)
        self._load_last_paths("synthesis", self.source_dir, self.output_dir)
        self._buttons(layout)

    def accept(self) -> None:
        source = Path(self.source_dir.text().strip())
        output_text = self.output_dir.text().strip()
        if not source.is_dir() or not output_text:
            AppDialog.information(
                "\u63d0\u793a",
                "\u8bf7\u9009\u62e9\u6709\u6548\u7684\u89c6\u9891\u5e27\u76ee\u5f55\u548c\u5408\u6210\u76ee\u6807\u76ee\u5f55\u3002",
                self,
            )
            return
        self._save_last_paths("synthesis", str(source), output_text)
        self.options = {
            "source_dir": source,
            "output_dir": Path(output_text),
            "format_name": self.format.currentData(),
        }
        super().accept()

class DatasetCompareDialog(_DirectoryDialog):
    """Model against dataset: which images does the model label differently?

    Read-only by construction. The only thing this operation writes is the
    report beside the dataset, which is also why the dataset directory is the
    last thing validated and the output path is never asked for: it follows
    from the dataset, and a second field would only be a way to get it wrong.
    """

    def __init__(self, parent=None, default_model: str = "", default_dataset: str = "") -> None:
        super().__init__("数据对比", parent)
        self.model_path = QLineEdit()
        self.dataset_dir = QLineEdit()
        self.threshold = QLineEdit("0.5")
        validator = QDoubleValidator(0.0, 1.0, 3, self.threshold)
        # The threshold is written with a dot in the report and in every other
        # place the app reads one, so it must not be parsed through a system
        # locale that writes 0,5.
        validator.setLocale(QLocale.c())
        self.threshold.setValidator(validator)
        layout = QVBoxLayout(self)
        set_content_margins(layout)
        layout.setSpacing(10)
        card = self._border_card(layout)
        form = configure_form(QFormLayout())
        form.addRow("模型路径", self._model_row(self.model_path))
        form.addRow("数据目录", self._path_row(self.dataset_dir, False))
        form.addRow("识别阈值", self.threshold)
        card.addLayout(form)
        self.model_path.setText(default_model or self._last_path("compare", "model_path"))
        self.dataset_dir.setText(default_dataset or self._last_path("compare", "dataset_dir"))
        self._buttons(layout)

    def accept(self) -> None:
        model = Path(self.model_path.text().strip())
        dataset = Path(self.dataset_dir.text().strip())
        if not model.is_file() or not dataset.is_dir() or not self.threshold.hasAcceptableInput():
            AppDialog.information(
                "提示",
                "请选择有效的 YOLO 模型文件和数据集目录，"
                "并填写 0 到 1 之间的识别阈值。",
                self,
            )
            return
        self._remember_path("compare", "model_path", str(model))
        self._remember_path("compare", "dataset_dir", str(dataset))
        self.options = {
            "model_path": model,
            "dataset_dir": dataset,
            "confidence_threshold": float(self.threshold.text()),
        }
        super().accept()


class DatasetCompareWorker(QObject):
    """Runs the comparison off the GUI thread and reports as it goes.

    A whole-dataset run is measured in hours, so ``cancel`` exists: the task
    list's stop button sets a flag the service reads between images, and the
    run ends without writing a report rather than writing a partial one.
    """

    progress = Signal(int, int)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, options: dict) -> None:
        super().__init__()
        self.options = options
        self.cancelled = False

    def run(self) -> None:
        try:
            from src.services.dataset_compare import compare_dataset

            report = compare_dataset(
                **self.options,
                progress_callback=lambda current, total: self.progress.emit(current, total),
                cancel_callback=lambda: self.cancelled,
            )
            self.completed.emit(report)
        except Exception as exc:
            self.failed.emit(str(exc))


class VideoToolsWorker(QObject):
    progress = Signal(str, int, int)
    completed = Signal(str, object)
    failed = Signal(str)

    def __init__(self, operation: str, options: dict, presets=None) -> None:
        super().__init__()
        self.operation = operation
        self.options = options
        self.presets = presets or []

    def run(self) -> None:
        try:
            from src.services.video_tools import extract_video_frames, synthesize_dataset
            callback = lambda current, total: self.progress.emit(self.operation, current, total)
            if self.operation == "frames":
                result = extract_video_frames(**self.options, progress_callback=callback)
            else:
                result = synthesize_dataset(**self.options, presets=self.presets, progress_callback=callback)
            self.completed.emit(self.operation, result)
        except Exception as exc:
            self.failed.emit(str(exc))