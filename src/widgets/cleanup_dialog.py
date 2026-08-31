from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from src.models.annotation import LabelPreset
from src.services.annotation_service import AnnotationService
from src.services.dataset_detector import DatasetDetector
from src.services.format_capabilities import CAPABILITIES, task_for_format
from .common_dialogs import AppDialog
from .form_layout import configure_buttons, configure_form, section_card, set_confirm_button, set_content_margins, size_buttons


class CleanupDialog(QDialog):
    """Scan a dataset and report files that have no usable annotations.

    An image is considered useless when:
    - it has no annotation file, or
    - its annotation file contains zero boxes/polygons/keypoints/obb rows.

    The dialog previews the count; actual deletion happens only after the
    user confirms, and files go to the recycle bin where possible.
    """

    def __init__(self, presets: list[LabelPreset], parent=None, default_source: str = "", language: str = "zh_CN") -> None:
        super().__init__(parent)
        self.setWindowTitle("清理数据")
        self.setMinimumWidth(480)
        self.english = language == "en_US"
        self.presets = list(presets)
        self._to_delete: list[Path] = []

        layout = QVBoxLayout(self); set_content_margins(layout); layout.setSpacing(10)

        # ---- source card (same language as the conversion dialog) ----------
        source_card = section_card(layout, "数据集合" if not self.english else "Dataset")
        source_form = configure_form(QFormLayout()); source_form.setVerticalSpacing(8)
        self.source_path = QLineEdit(default_source)
        source_form.addRow("目录" if not self.english else "Directory", self._path_row())
        self.source_format = QComboBox()
        self.source_format.addItem("COCO", "coco"); self.source_format.addItem("YOLO", "yolo"); self.source_format.addItem("Pascal VOC", "voc")
        source_form.addRow("格式" if not self.english else "Format", self.source_format)
        source_card.addLayout(source_form)

        # ---- scan + report ---------------------------------------------------
        self.scan_button = QPushButton("扫描" if not self.english else "Scan")
        self.scan_button.clicked.connect(self._scan)
        source_card.addWidget(self.scan_button)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(
            "QLabel { background: #2A2C31; border: 1px solid #3E424A; border-radius: 5px; "
            "padding: 8px 10px; color: #B8C7E6; font-size: 12px; }"
        )
        self.result_label.setMinimumHeight(50)
        layout.addWidget(self.result_label)

        # ---- buttons ---------------------------------------------------
        buttons = configure_buttons(QHBoxLayout()); buttons.addStretch()
        self.cancel_button = QPushButton("取消" if not self.english else "Cancel")
        self.confirm_button = QPushButton("清理" if not self.english else "Clean")
        self.confirm_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.reject)
        self.confirm_button.clicked.connect(self._clean)
        buttons.addWidget(self.cancel_button); buttons.addWidget(self.confirm_button)
        size_buttons(self.cancel_button, self.confirm_button)
        self.confirm_button.setFixedHeight(30)
        set_confirm_button(self.scan_button)  # Enter triggers scan first

        self._detect_source(default_source)

    def _path_row(self) -> QWidget:
        row = QWidget(); layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(8)
        layout.addWidget(self.source_path)
        browse = QPushButton("浏览" if not self.english else "Browse")
        browse.setFixedHeight(30)
        browse.clicked.connect(self._choose)
        layout.addWidget(browse)
        return row

    def _choose(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录" if not self.english else "Choose dataset", self.source_path.text())
        if path:
            self.source_path.setText(path)
            self._detect_source(path)

    def _detect_source(self, text: str) -> None:
        source = Path(text.strip()) if text.strip() else None
        if source is None or not source.is_dir():
            return
        try:
            detected = DatasetDetector.detect(source)
            index = self.source_format.findData(detected.format_name)
            if index >= 0:
                self.source_format.setCurrentIndex(index)
        except ValueError:
            pass

    def _scan(self) -> None:
        source = Path(self.source_path.text().strip())
        if not source.is_dir():
            AppDialog.information("提示", "目录不存在。", self)
            return
        try:
            detected = DatasetDetector.detect(source)
        except ValueError:
            AppDialog.information("提示", "无法识别该数据集格式。", self)
            return
        self.source_format.setCurrentIndex(max(0, self.source_format.findData(detected.format_name)))

        format_name = detected.format_name
        settings = type("S", (), {
            "annotation_format": format_name,
            "dataset_task": detected.task_name,
            "label_presets": self.presets,
            "image_dir": detected.image_dir,
        })()
        service = AnnotationService()

        from src.services.image_service import ImageService, SUPPORTED_IMAGE_EXTENSIONS
        images = sorted(path for path in detected.image_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
        useless: list[Path] = []
        for image in images:
            record = type("R", (), {"path": image, "width": 0, "height": 0, "file_format": "", "file_size": 0})()
            try:
                result = service.load(image, detected.annotation_dir, settings)
                if result.error or not result.annotations:
                    useless.append(image)
            except Exception:
                useless.append(image)

        self._to_delete = useless
        total = len(images)
        useful = total - len(useless)
        if useless:
            self.result_label.setText(
                f"共 {total} 张图片，{useful} 张有有效标注。以下 {len(useless)} 张没有标注或标注为空，可清理：\n"
                + "\n".join(str(p.relative_to(source)) for p in useless[:12])
                + ("\n…" if len(useless) > 12 else "")
            )
            self.confirm_button.setEnabled(True)
            self.scan_button.setDefault(False)
            set_confirm_button(self.confirm_button)
        else:
            self.result_label.setText(f"共 {total} 张图片，全部有有效标注，无需清理。")
            self.confirm_button.setEnabled(False)

    def _clean(self) -> None:
        if not self._to_delete:
            return
        count = len(self._to_delete)
        if not AppDialog.question("提示", f"确认清理 {count} 个文件？此操作不可恢复。", self):
            return
        deleted = 0
        for path in self._to_delete:
            try:
                # Best effort: try send2trash, fallback to unlink
                try:
                    from send2trash import send2trash
                    send2trash(str(path))
                except ImportError:
                    path.unlink(missing_ok=True)
                deleted += 1
            except OSError:
                pass
        AppDialog.information("提示", f"已清理 {deleted}/{count} 个文件。", self)
        self.accept()

    @property
    def cleaned_source(self) -> str:
        return self.source_path.text().strip()
