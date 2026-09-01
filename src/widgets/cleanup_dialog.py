from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from src.models.annotation import LabelPreset
from src.services.annotation_service import AnnotationService
from src.services.coco_store import CocoAnnotationStore
from src.services.dataset_detector import DatasetDetector
from .common_dialogs import AppDialog
from .form_layout import configure_buttons, configure_form, section_card, set_confirm_button, set_content_margins, size_buttons


class CleanupDialog(QDialog):
    """Remove images with no usable annotations and keep image/annotation
    files in sync.

    Cleanup rules (all formats):
    - an image whose annotation file has zero boxes/polygons/keypoints/obb
      rows is deleted together with its annotation file
    - an annotation file whose image no longer exists (deleted outside the
      app) is deleted as well
    For COCO the same rules apply to the JSON: image records (and their
    annotations) are removed when the image file is gone or unannotated.

    Every action is logged into a read-only, selectable text area below the
    start button so the annotator can copy the report.
    """

    def __init__(self, presets: list[LabelPreset], parent=None, default_source: str = "", language: str = "zh_CN") -> None:
        super().__init__(parent)
        self.setWindowTitle("清理数据")
        self.setMinimumWidth(560)
        self.english = language == "en_US"
        self.presets = list(presets)
        self._to_delete_images: list[Path] = []
        self._to_delete_annotations: list[Path] = []
        self._format_name = ""

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

        # ---- scan result module ----------------------------------------------
        result_card = section_card(layout, "扫描结果" if not self.english else "Scan Result")
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(
            "QLabel { background: #2A2C31; border: 1px solid #3E424A; border-radius: 5px; "
            "padding: 8px 10px; color: #B8C7E6; font-size: 12px; }"
        )
        self.result_label.setMinimumHeight(50)
        result_card.addWidget(self.result_label)

        # ---- cleanup operation module ----------------------------------------
        cleanup_card = section_card(layout, "清理操作" if not self.english else "Cleanup")

        # risk warning above the action buttons
        self.warning_label = QLabel(
            "⚠ 危险操作：清理会永久删除以下文件（优先移入回收站，但无法保证恢复）：\n"
            "无标注的图片、与之对应的标注文件、以及图片已不存在的孤立标注文件；COCO 会同时改写 annotations.json。请先确认扫描结果，建议提前备份数据集。"
            if not self.english else
            "⚠ Dangerous: cleanup permanently deletes unannotated images, their "
            "annotation files, and orphan annotation files; COCO also rewrites "
            "annotations.json. Review the scan report first and back up the dataset."
        )
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(
            "QLabel { background: #3A2A24; border: 1px solid #8A4B3A; border-radius: 5px; "
            "padding: 8px 10px; color: #FFB08A; font-size: 12px; font-weight: 600; }"
        )
        cleanup_card.addWidget(self.warning_label)

        buttons = configure_buttons(QHBoxLayout()); buttons.addStretch()
        self.cancel_button = QPushButton("取消" if not self.english else "Cancel")
        self.confirm_button = QPushButton("开始清理" if not self.english else "Start Cleanup")
        self.confirm_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.reject)
        self.confirm_button.clicked.connect(self._clean)
        buttons.addWidget(self.cancel_button); buttons.addWidget(self.confirm_button)
        size_buttons(self.cancel_button, self.confirm_button)
        self.confirm_button.setFixedHeight(30)
        cleanup_card.addLayout(buttons)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText(
            "清理日志：每条操作的详细记录，可直接用鼠标选中复制。" if not self.english
            else "Cleanup log: select any text to copy it."
        )
        self.log_view.setStyleSheet(
            "QPlainTextEdit { background: #1E2024; color: #C9D1D9; border: 1px solid #3E424A; "
            "border-radius: 5px; padding: 6px; font-family: Consolas, 'Courier New', monospace; "
            "font-size: 12px; }"
        )
        self.log_view.setMinimumHeight(160)
        cleanup_card.addWidget(self.log_view, 1)

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

    def _annotation_file_for(self, image: Path) -> Path | None:
        """Mirror an image path onto its annotation file (YOLO .txt / VOC .xml)."""
        if self._format_name not in ("yolo", "voc"):
            return None
        try:
            relative = image.relative_to(self._image_dir)
        except ValueError:
            return None
        suffix = ".txt" if self._format_name == "yolo" else ".xml"
        candidate = self._annotation_dir / relative.with_suffix(suffix)
        return candidate if candidate.is_file() else None

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
        self._format_name = detected.format_name
        self._image_dir = detected.image_dir
        self._annotation_dir = detected.annotation_dir

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

        # Annotation files whose image no longer exists (keeps folders in
        # sync when images were deleted outside the app).
        orphans: list[Path] = []
        if format_name in ("yolo", "voc"):
            suffix = ".txt" if format_name == "yolo" else ".xml"
            for annotation_file in sorted(detected.annotation_dir.rglob(f"*{suffix}")):
                if format_name == "yolo" and annotation_file.name == "classes.txt":
                    continue
                try:
                    relative = annotation_file.relative_to(detected.annotation_dir)
                except ValueError:
                    continue
                # Path.glob with ".*" hits a pathlib parse bug on Windows;
                # match by stem over the image tree instead.
                image_candidates = [
                    p for p in detected.image_dir.rglob("*")
                    if p.is_file()
                    and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                    and p.name.lower().startswith(relative.stem.lower() + ".")
                ]
                if not image_candidates:
                    orphans.append(annotation_file)

        self._to_delete_images = useless
        self._to_delete_annotations = orphans
        total = len(images)
        useful = total - len(useless)
        if useless or orphans:
            lines = [f"共 {total} 张图片，{useful} 张有有效标注。"]
            if useless:
                lines.append(f"以下 {len(useless)} 张图片没有标注或标注为空，将连同标注文件一起删除：")
                lines += [f"  - {p.relative_to(source)}" for p in useless[:10]]
                if len(useless) > 10:
                    lines.append(f"  …（其余 {len(useless) - 10} 张略）")
            if orphans:
                lines.append(f"以下 {len(orphans)} 个标注文件对应的图片已不存在：")
                lines += [f"  - {p.relative_to(source)}" for p in orphans[:10]]
                if len(orphans) > 10:
                    lines.append(f"  …（其余 {len(orphans) - 10} 个略）")
            self.result_label.setText("\n".join(lines))
            self.confirm_button.setEnabled(True)
            self.scan_button.setDefault(False)
            set_confirm_button(self.confirm_button)
        else:
            self.result_label.setText(f"共 {total} 张图片，全部有有效标注，标注文件与图片一一对应，无需清理。")
            self.confirm_button.setEnabled(False)

    def _trash(self, path: Path) -> bool:
        """Recycle bin first, plain unlink as fallback; returns success."""
        try:
            try:
                from send2trash import send2trash
                send2trash(str(path))
            except ImportError:
                path.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def _sync_coco_json(self, removed_image_names: set[str]) -> None:
        """Remove JSON image records and their annotations for deleted files."""
        try:
            store = CocoAnnotationStore(self._annotation_dir)
            document = store.read_document()
            removed_ids = {
                int(item.get("id"))
                for item in document.get("images", [])
                if Path(str(item.get("file_name", ""))).name in removed_image_names
            }
            if not removed_ids:
                return
            document["images"] = [
                item for item in document.get("images", [])
                if int(item.get("id")) not in removed_ids
            ]
            document["annotations"] = [
                item for item in document.get("annotations", [])
                if int(item.get("image_id")) not in removed_ids
            ]
            store.replace_document(document)
            store.export_json()
        except (OSError, ValueError):
            pass

    def _clean(self) -> None:
        if not (self._to_delete_images or self._to_delete_annotations):
            return
        count = len(self._to_delete_images) + len(self._to_delete_annotations)
        if not AppDialog.question("提示", f"确认清理 {count} 个文件？此操作不可恢复。", self):
            return

        source = Path(self.source_path.text().strip())
        log: list[str] = []
        deleted = 0

        # 1) unannotated images, then their mirrored annotation files
        for path in self._to_delete_images:
            if self._trash(path):
                deleted += 1
                log.append(f"删除图片: {path.relative_to(source)}")
            else:
                log.append(f"失败: {path.relative_to(source)}")
        for path in self._to_delete_images:
            annotation_file = self._annotation_file_for(path)
            if annotation_file is not None and annotation_file.exists():
                if self._trash(annotation_file):
                    deleted += 1
                    log.append(f"同步删除标注: {annotation_file.relative_to(source)}")
                else:
                    log.append(f"失败: {annotation_file.relative_to(source)}")

        # 2) annotation files whose image is gone
        for path in self._to_delete_annotations:
            if self._trash(path):
                deleted += 1
                log.append(f"删除孤立标注: {path.relative_to(source)}")
            else:
                log.append(f"失败: {path.relative_to(source)}")

        # 3) COCO: drop JSON records for the removed images
        if self._format_name == "coco":
            removed = {path.name for path in self._to_delete_images}
            before = len(log)
            self._sync_coco_json(removed)
            if len(log) == before and removed:
                log.append(f"已从 annotations.json 移除 {len(removed)} 张图片的记录")

        self.log_view.setPlainText("\n".join(log) if log else "未执行任何删除。")
        self.log_view.verticalScrollBar().setValue(0)
        AppDialog.information("提示", f"已清理 {deleted}/{count} 个文件，详见下方日志。", self)
        self.accept()

    @property
    def cleaned_source(self) -> str:
        return self.source_path.text().strip()
