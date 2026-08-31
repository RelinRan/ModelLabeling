from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from src.models.annotation import ShapeType
from src.services.dataset_initializer import DatasetInitializer
from src.services.format_capabilities import CAPABILITIES, task_for_format
from .form_layout import configure_buttons, configure_form, section_card, set_confirm_button
from .numeric_stepper import NumericStepper


TASK_OPTIONS = (
    ("yolo_detection", "YOLO 检测（矩形框）", "YOLO Detection (boxes)"),
    ("yolo_segmentation", "YOLO 分割（多边形）", "YOLO Segmentation (polygons)"),
    ("yolo_pose", "YOLO 关键点（人体/姿态）", "YOLO Pose (keypoints)"),
    ("yolo_obb", "YOLO 旋转框（倾斜目标）", "YOLO OBB (rotated boxes)"),
    ("voc", "Pascal VOC（矩形框）", "Pascal VOC (boxes)"),
    ("coco", "COCO（框/多边形/关键点）", "COCO (boxes/polygons/keypoints)"),
)


class DatasetInitDialog(QDialog):
    """Create a new dataset inside a workspace, like an IDE project.

    The workspace holds many datasets; the source image folder is only
    material whose images are copied into workspace/<name>/images.
    """

    initialized = Signal(object)  # ("created", InitializedDataset)

    def __init__(self, parent=None, language: str = "zh_CN", initial_workspace: str = "") -> None:
        super().__init__(parent)
        self.english = language == "en_US"
        self.setWindowTitle("新建数据集" if not self.english else "New Dataset")
        self.resize(588, 404)
        self.result_info = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        # ---- card: basic info ------------------------------------------
        basic_card = section_card(layout, "基本信息" if not self.english else "Basics")
        basic = configure_form(QFormLayout())
        basic.setVerticalSpacing(8)
        self.workspace_edit = QLineEdit(initial_workspace)
        self.workspace_edit.setPlaceholderText(r"如 E:\Datasets" if not self.english else r"e.g. E:\Datasets")
        basic.addRow("工作空间" if not self.english else "Workspace", self._path_row(self.workspace_edit, self._choose_workspace))

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("如 model-dataset" if not self.english else "e.g. model-dataset")
        basic.addRow("数据集合" if not self.english else "Name", self.name_edit)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText(
            "可选：图片目录、数据集目录"
            if not self.english
            else "Optional: image folder or dataset folder"
        )
        basic.addRow("数据来源" if not self.english else "Source", self._path_row(self.source_edit, self._choose_source))
        basic_card.addLayout(basic)

        # ---- card: annotation settings, one field per row ---------------
        annotation_card = section_card(layout, "标注配置" if not self.english else "Annotation")
        self.annotation_form = annotation_form = configure_form(QFormLayout())
        annotation_form.setVerticalSpacing(8)
        self.task_combo = QComboBox()
        for value, zh, en in TASK_OPTIONS:
            self.task_combo.addItem(en if self.english else zh, value)
        self.task_combo.currentIndexChanged.connect(self._refresh_hint)
        annotation_form.addRow("标注格式" if not self.english else "Format", self.task_combo)

        self.keypoint_count = NumericStepper(17, 1, 135, 1)
        self.keypoint_count.setEnabled(False)
        annotation_form.addRow("关键点数" if not self.english else "Keypoints", self.keypoint_count)

        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("可选：逗号分隔，如 person, head, car" if not self.english else "Optional: comma separated, e.g. person, head, car")
        annotation_form.addRow("标签类别" if not self.english else "Classes", self.classes_edit)
        annotation_card.addLayout(annotation_form)

        # ---- structure preview inside the annotation card ---------------
        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #9aa0a8; border: none; background: transparent; font-size: 13px; padding: 0;")
        annotation_card.addWidget(self.hint_label)

        layout.addSpacing(4)

        # ---- live status bar ------------------------------------------
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.RichText)
        self.status_label.setStyleSheet(
            "QLabel { background: #2A2C31; border: 1px solid #3E424A; border-radius: 5px; "
            "padding: 6px 10px; color: #B8C7E6; }"
        )
        self.status_label.setMinimumHeight(34)
        layout.addWidget(self.status_label)

        # ---- buttons ---------------------------------------------------
        buttons = configure_buttons(QHBoxLayout())
        buttons.addSpacing(0)
        buttons.addStretch()
        self.cancel_button = QPushButton("取消" if not self.english else "Cancel")
        self.confirm_button = QPushButton("创建" if not self.english else "Create")
        self.cancel_button.clicked.connect(self.reject)
        self.confirm_button.clicked.connect(self._create)
        buttons.addWidget(self.cancel_button); buttons.addWidget(self.confirm_button)
        layout.addLayout(buttons)
        self.cancel_button.setFixedHeight(30)
        self.confirm_button.setFixedHeight(30)
        set_confirm_button(self.confirm_button)

        self.workspace_edit.textChanged.connect(self._refresh_status)
        self.name_edit.textChanged.connect(self._refresh_status)
        self.source_edit.textChanged.connect(self._refresh_status)
        self._refresh_status()
        self._refresh_hint()


    def _path_row(self, editor: QLineEdit, handler) -> QWidget:
        row = QWidget(); row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0); row_layout.setSpacing(8)
        editor.textChanged.connect(self._refresh_status)
        browse = QPushButton("浏览" if not self.english else "Browse")
        browse.setFixedHeight(30)
        browse.clicked.connect(handler)
        row_layout.addWidget(editor, 1); row_layout.addWidget(browse)
        return row

    def _choose_workspace(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择工作空间" if not self.english else "Choose workspace", self.workspace_edit.text())
        if path:
            self.workspace_edit.setText(path)

    def _choose_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据来源" if not self.english else "Choose source folder", self.source_edit.text())
        if path:
            self.source_edit.setText(path)

    def _set_status(self, message: str, tone: str = "info") -> None:
        colors = {
            "info": ("#2A2C31", "#3E424A", "#B8C7E6"),
            "ready": ("#25303F", "#33507E", "#B8D4F0"),
            "error": ("#3A2A2C", "#6E3B41", "#F0B9BE"),
        }[tone]
        background, border, color = colors
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            f"QLabel {{ background: {background}; border: 1px solid {border}; "
            f"border-radius: 5px; padding: 6px 10px; color: {color}; }}"
        )

    def _refresh_status(self) -> None:
        english = self.english
        workspace = self.workspace_edit.text().strip()
        name = self.name_edit.text().strip()
        source = self.source_edit.text().strip()
        if not workspace and not name and not source:
            self._set_status(
                "数据集将创建在工作空间中，图片仅作为素材复制进来" if not english
                else "The dataset is created inside the workspace; images are copied in as material"
            )
            return
        if not workspace:
            self._set_status("请选择工作空间" if not english else "Choose a workspace", "error")
            return
        error = DatasetInitializer.validate_name(name) if name else ("数据集名称不能为空" if not english else "Dataset name is required")
        if name and error is None:
            target = Path(workspace) / name
            if target.exists() and any(target.iterdir()):
                error = f"已存在同名数据集: {target}" if not english else f"Already exists: {target}"
        if error:
            self._set_status(error, "error")
            return
        if not source:
            self._set_status(
                f"✓ 将在 {workspace} 下创建 {name}（空数据集，可稍后放入图片）" if not english
                else f"✓ {name} will be created under {workspace} (empty; add images later)",
                "ready",
            )
            return
        source_path = Path(source)
        if not source_path.is_dir():
            self._set_status("数据来源目录不存在" if not english else "Source folder does not exist", "error")
            return
        count = DatasetInitializer.count_images(source_path)
        from src.services.dataset_detector import DatasetDetector
        try:
            detected = DatasetDetector.detect(source_path, allow_plain_images=False)
            detected_text = (
                f"，源是 {detected.format_name} 数据集，标注将一并导入"
                if not english else f"; source is a {detected.format_name} dataset, annotations will be imported"
            )
        except ValueError:
            detected_text = ""
        self._set_status(
            (f"✓ 将创建 {name}，复制 {count} 张图片" if not english else f"✓ {name} will be created with {count} images")
            + detected_text,
            "ready",
        )

    def _refresh_hint(self) -> None:
        task = str(self.task_combo.currentData() or "")
        english = self.english
        structure = DatasetInitializer.structure_summary(task)
        self.hint_label.setText((("Data structure: " if english else "数据结构：")) + structure)
        # Only formats whose capability matrix supports keypoints get the
        # keypoint-count setting; everything else hides the row entirely.
        capability_task = task_for_format(task, task)
        show_keypoints = ShapeType.KEYPOINT in CAPABILITIES[capability_task].shapes
        self.keypoint_count.setEnabled(show_keypoints)
        for row in range(self.annotation_form.rowCount()):
            field = self.annotation_form.itemAt(row, QFormLayout.ItemRole.FieldRole)
            if field is not None and field.widget() is self.keypoint_count:
                self.annotation_form.setRowVisible(row, show_keypoints)
                break

    def _create(self) -> None:
        workspace = self.workspace_edit.text().strip()
        name = self.name_edit.text().strip()
        source = self.source_edit.text().strip()
        if not workspace or DatasetInitializer.validate_name(name):
            self._refresh_status()
            return
        names = [part.strip() for part in self.classes_edit.text().split(",") if part.strip()]
        try:
            info = DatasetInitializer.create_in_workspace(
                Path(workspace), name, str(self.task_combo.currentData()),
                names, self.keypoint_count.value(),
                Path(source) if source else None,
            )
        except ValueError as exc:
            self._set_status(str(exc), "error")
            return
        self.result_info = ("created", info)
        self.initialized.emit(self.result_info)
        self.accept()
