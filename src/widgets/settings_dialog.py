from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from src.models.annotation import ShapeType
from src.models.project import ProjectSettings
from src.services.display_names import format_options, task_options
from src.services.format_capabilities import CAPABILITIES, task_for_format
from .canvas_view import CanvasView
from .i18n import LANGUAGES
from .form_layout import BUTTON_TOP_SPACING, configure_buttons, configure_form, section_card, set_confirm_button, set_content_margins, size_buttons
from .numeric_stepper import NumericStepper


class SettingsDialog(QDialog):
    settingsChanged = Signal(object)

    def __init__(self, settings: ProjectSettings, parent=None) -> None:
        super().__init__(parent)
        self.settings = ProjectSettings.from_dict(settings.to_dict())
        self.english = self.settings.language == "en_US"
        self.setObjectName("settingsDialog"); self.resize(500, 335); self.setMinimumWidth(470); self.setWindowTitle("Application Settings" if self.english else "应用设置")
        self.image_dir = QLineEdit(str(self.settings.image_dir or "")); self.annotation_dir = QLineEdit(str(self.settings.annotation_dir or "")); self.onnx_model = QLineEdit(str(self.settings.onnx_model_path or ""))
        self.format = QComboBox()
        for label, value in format_options("en_US" if self.english else "zh_CN"):
            self.format.addItem(label, value)
        self.format.setCurrentIndex(max(0, self.format.findData(self.settings.annotation_format)))
        self.task = QComboBox()
        self.format.currentIndexChanged.connect(self._refresh_task_options)
        self._refresh_task_options()
        self.language_combo = QComboBox()
        for value, label in LANGUAGES.items(): self.language_combo.addItem(label, value)
        self.language_combo.setCurrentIndex(max(0, self.language_combo.findData(self.settings.language)))
        self.reopen_combo = QComboBox(); self.reopen_combo.addItem("启动时恢复上次数据集" if not self.english else "Reopen last dataset on start", True); self.reopen_combo.addItem("启动时空画布" if not self.english else "Start with an empty canvas", False); self.reopen_combo.setCurrentIndex(0 if self.settings.reopen_last_dataset else 1)
        self.line_width = NumericStepper(self.settings.line_width, 1, 12, 1)
        self.text_size = NumericStepper(self.settings.text_size, 6, 48, 1)
        self.auto_save = QComboBox(); self.auto_save.addItem("Auto save" if self.english else "\u81ea\u52a8\u4fdd\u5b58", True); self.auto_save.addItem("Manual save" if self.english else "\u624b\u52a8\u4fdd\u5b58", False); self.auto_save.setCurrentIndex(0 if self.settings.auto_save else 1)
        self.shape_combo = QComboBox()
        for shape in CanvasView.METHOD_ORDER:
            self.shape_combo.addItem(CanvasView.method_label(shape, "en_US" if self.english else "zh_CN"), shape)
        self.shape_combo.setCurrentIndex(max(0, self.shape_combo.findData(self.settings.enabled_shapes[0])))
        self.task.currentIndexChanged.connect(self._refresh_shape_options)
        if hasattr(self, "shape_combo"):
            self._refresh_shape_options()
        self.input_width = NumericStepper(self.settings.input_width, 32, 2048, 1)
        self.input_height = NumericStepper(self.settings.input_height, 32, 2048, 1)
        self.input_size_row = QWidget(); self.input_size_layout = QHBoxLayout(self.input_size_row); self.input_size_layout.setContentsMargins(0, 0, 0, 0); self.input_size_layout.setSpacing(6); self.input_size_layout.addWidget(self.input_width, 1); self.input_size_layout.addWidget(QLabel("x")); self.input_size_layout.addWidget(self.input_height, 1)
        self.confidence = NumericStepper(self.settings.confidence_threshold, 0, 1, 0.05, 2)
        self.nms = NumericStepper(self.settings.nms_threshold, 0, 1, 0.05, 2)
        self.categories = QListWidget(); self.categories.setObjectName("settingsCategories"); self.categories.setFixedWidth(95); self.categories.setFrameShape(QListWidget.Shape.NoFrame); self.categories.setLineWidth(0); self.categories.setMidLineWidth(0); self.categories.addItems(["Dataset", "Display", "Auto Label", "General"] if self.english else ["数据集", "标注显示", "自动标注", "通用设置"])
        self.pages = QStackedWidget(); self.pages.addWidget(self._dataset_page()); self.pages.addWidget(self._annotation_page()); self.pages.addWidget(self._auto_page()); self.pages.addWidget(self._general_page())
        self.categories.currentRowChanged.connect(self.pages.setCurrentIndex); self.categories.setCurrentRow(0)
        layout = QVBoxLayout(self); set_content_margins(layout); layout.setSpacing(0); content = QHBoxLayout(); content.setContentsMargins(0, 0, 0, 0); content.setSpacing(20); content.addWidget(self.categories); content.addWidget(self.pages, 1); layout.addLayout(content)
        layout.addSpacing(26)
        buttons = configure_buttons(QHBoxLayout()); buttons.addStretch(); self.cancel_button = QPushButton("Cancel" if self.english else "\u53d6\u6d88"); self.confirm_button = QPushButton("Confirm" if self.english else "\u786e\u8ba4"); self.cancel_button.clicked.connect(self.reject); self.confirm_button.clicked.connect(self.accept); buttons.addWidget(self.cancel_button); buttons.addWidget(self.confirm_button); layout.addLayout(buttons)
        size_buttons(self.cancel_button, self.confirm_button)
        set_confirm_button(self.confirm_button)

    def _page(self, title: str, rows: list[tuple[str, QWidget]]) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(0, 0, 4, 0); layout.setSpacing(0)
        card = section_card(layout, title)
        form = configure_form(QFormLayout())
        for label, widget in rows: form.addRow(label, widget)
        card.addLayout(form); layout.addStretch(); return page

    def _dataset_page(self) -> QWidget: return self._page("Dataset" if self.english else "\u6570\u636e\u96c6", [("Image path" if self.english else "\u56fe\u7247\u8def\u5f84", self._path_row(self.image_dir, self._choose_image_dir)), ("Annotation path" if self.english else "\u6807\u6ce8\u8def\u5f84", self._path_row(self.annotation_dir, self._choose_annotation_dir)), ("Format" if self.english else "\u6807\u6ce8\u683c\u5f0f", self.format), ("Task" if self.english else "\u4efb\u52a1\u683c\u5f0f", self.task)])
    def _annotation_page(self) -> QWidget: return self._page("Display" if self.english else "\u6807\u6ce8\u663e\u793a", [("Annotation method" if self.english else "\u6807\u6ce8\u65b9\u5f0f", self.shape_combo), ("Line width" if self.english else "\u7ebf\u6761\u7c97\u7ec6", self.line_width), ("Text size" if self.english else "\u6587\u5b57\u5927\u5c0f", self.text_size)])
    def _auto_page(self) -> QWidget: return self._page("Auto Label" if self.english else "\u81ea\u52a8\u6807\u6ce8", [("YOLO model" if self.english else "YOLO模型", self._path_row(self.onnx_model, self._choose_model)), ("Input size" if self.english else "\u8f93\u5165\u5c3a\u5bf8", self.input_size_row), ("Confidence" if self.english else "\u7f6e\u4fe1\u5ea6\u9608\u503c", self.confidence), ("NMS threshold" if self.english else "NMS \u9608\u503c", self.nms)])
    def _general_page(self) -> QWidget: return self._page("General" if self.english else "\u901a\u7528\u8bbe\u7f6e", [("Application language" if self.english else "\u5e94\u7528\u8bed\u8a00", self.language_combo), ("Startup" if self.english else "\u542f\u52a8", self.reopen_combo), ("Save mode" if self.english else "\u6807\u6ce8\u4fdd\u5b58", self.auto_save)])
    def _path_row(self, editor: QLineEdit, handler) -> QWidget:
        row = QWidget(); layout = QHBoxLayout(row); layout.setContentsMargins(0, 0, 0, 0); layout.addWidget(editor); button = QPushButton("Browse" if self.english else "\u6d4f\u89c8"); button.clicked.connect(handler); layout.addWidget(button); return row
    def _choose_image_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose image folder" if self.english else "选择图片文件夹", self.image_dir.text()); self.image_dir.setText(path) if path else None
    def _choose_annotation_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose annotation folder" if self.english else "选择标注文件夹", self.annotation_dir.text()); self.annotation_dir.setText(path) if path else None
    def _choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose ONNX model" if self.english else "选择 ONNX 模型", self.onnx_model.text(), "ONNX Model (*.onnx)"); self.onnx_model.setText(path) if path else None

    def _refresh_task_options(self) -> None:
        current = self.settings.dataset_task or ""
        language = "en_US" if self.english else "zh_CN"
        self.task.blockSignals(True)
        self.task.clear()
        for label, value in task_options(self.format.currentData(), language):
            self.task.addItem(label, value)
        index = self.task.findData(current)
        self.task.setCurrentIndex(index if index >= 0 else 0)
        self.task.blockSignals(False)
        if hasattr(self, "shape_combo"):
            self._refresh_shape_options()

    def _refresh_shape_options(self) -> None:
        # Methods the current task cannot save are hidden entirely rather
        # than shown disabled: the dropdown must only offer valid choices.
        task = str(self.task.currentData() or "")
        supported = CAPABILITIES.get(task_for_format(str(self.format.currentData()), task), CAPABILITIES[task_for_format("yolo")]).shapes
        current = self.shape_combo.currentData()
        self.shape_combo.blockSignals(True)
        self.shape_combo.clear()
        for shape in CanvasView.METHOD_ORDER:
            if shape in supported:
                self.shape_combo.addItem(CanvasView.method_label(shape, "en_US" if self.english else "zh_CN"), shape)
        index = self.shape_combo.findData(current) if current in supported else 0
        self.shape_combo.setCurrentIndex(max(0, index))
        self.shape_combo.blockSignals(False)

    def apply(self) -> ProjectSettings:
        self.settings.image_dir = Path(self.image_dir.text()) if self.image_dir.text() else None; self.settings.annotation_dir = Path(self.annotation_dir.text()) if self.annotation_dir.text() else None; self.settings.onnx_model_path = Path(self.onnx_model.text()) if self.onnx_model.text() else None
        self.settings.annotation_format = str(self.format.currentData()); self.settings.dataset_task = str(self.task.currentData()); self.settings.language = self.language_combo.currentData(); self.settings.reopen_last_dataset = bool(self.reopen_combo.currentData()); self.settings.enabled_shapes = [ShapeType(self.shape_combo.currentData())]; self.settings.line_width = self.line_width.value(); self.settings.text_size = self.text_size.value(); self.settings.auto_save = bool(self.auto_save.currentData()); self.settings.input_width = self.input_width.value(); self.settings.input_height = self.input_height.value(); self.settings.input_size = self.input_width.value(); self.settings.confidence_threshold = self.confidence.value(); self.settings.nms_threshold = self.nms.value()
        return ProjectSettings.from_dict(self.settings.to_dict())

    def accept(self) -> None:
        # Keep the dialog's public settings object synchronized with the
        # values emitted to MainWindow. MainWindow reads dialog.settings after
        # exec() returns, so leaving this as the old snapshot silently drops
        # shape changes such as Rectangle -> Square.
        self.settings = self.apply()
        self.settingsChanged.emit(self.settings)
        super().accept()
