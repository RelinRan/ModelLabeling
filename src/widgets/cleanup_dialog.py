from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget

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

    Scanning runs on a background thread so large datasets never freeze the
    dialog; the scan button shows live progress (xx%), and every action is
    logged into a read-only, selectable text area below the start button.
    """

    scan_progress = Signal(int, int, int)
    scan_finished = Signal(object)

    def __init__(self, presets: list[LabelPreset], parent=None, default_source: str = "", language: str = "zh_CN") -> None:
        super().__init__(parent)
        self.setWindowTitle("清理数据")
        self.setMinimumWidth(560)
        self.english = language == "en_US"
        self.presets = list(presets)
        self._to_delete_images: list[Path] = []
        self._to_delete_annotations: list[Path] = []
        self._format_name = ""
        # Scan totals feed the post-cleanup summary line.
        self._scan_total = 0
        self._scan_useful = 0

        layout = QVBoxLayout(self); set_content_margins(layout); layout.setSpacing(10)

        # ---- source card (same language as the conversion dialog) ----------
        source_card = section_card(layout, "数据集合" if not self.english else "Dataset")
        source_form = configure_form(QFormLayout()); source_form.setVerticalSpacing(8)
        self.source_path = QLineEdit(default_source)
        self.source_path.textChanged.connect(self._on_source_changed)
        source_form.addRow("目录" if not self.english else "Directory", self._path_row())
        self.source_format = QComboBox()
        self.source_format.addItem("COCO", "coco"); self.source_format.addItem("YOLO", "yolo"); self.source_format.addItem("Pascal VOC", "voc")
        source_form.addRow("格式" if not self.english else "Format", self.source_format)
        source_card.addLayout(source_form)

        # ---- scan + report ---------------------------------------------------
        self.scan_button = QPushButton("开始扫描" if not self.english else "Start Scan")
        self.scan_button.clicked.connect(self._start_scan)
        self.scan_button.setFixedHeight(30)
        source_card.addWidget(self.scan_button)
        self.scan_progress.connect(self._on_scan_progress)
        self.scan_finished.connect(self._on_scan_finished)
        # ---- scan result module ----------------------------------------------
        result_card = section_card(layout, "扫描结果" if not self.english else "Scan Result")
        self.result_label = QPlainTextEdit()
        self.result_label.setReadOnly(True)
        # Keep text selectable and copyable (mouse, keyboard, context menu).
        self.result_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.result_label.setPlaceholderText(
            "点击「开始扫描」后在此显示扫描结果。" if not self.english
            else "Start a scan to see the report here."
        )
        self.result_label.setStyleSheet(
            "QPlainTextEdit { background: #2A2C31; color: #B8C7E6; border: 1px solid #3E424A; "
            "border-radius: 5px; padding: 6px; font-family: Consolas, 'Courier New', monospace; "
            "font-size: 12px; }"
        )
        # Fixed height so the pane never absorbs spare layout height and the
        # card hugs its content; the layout uses 100 instead of the widget's
        # larger sizeHint.
        self.result_label.setFixedHeight(100)
        result_card.addWidget(self.result_label)

        # ---- cleanup operation module ----------------------------------------
        # The risk warning follows the section title directly.
        self.warning_label = QLabel(
            "[删除不可恢复，请先备份数据]" if not self.english
            else "[Deletions are permanent; back up the dataset first]"
        )
        self.warning_label.setStyleSheet(
            "QLabel { background: transparent; border: none; color: #FFB08A; "
            "font-size: 12px; font-weight: 600; }"
        )
        cleanup_card = section_card(
            layout, "清理操作" if not self.english else "Cleanup",
            badge=self.warning_label, badge_after_title=True,
        )

        buttons = configure_buttons(QHBoxLayout())
        # The start-cleanup button keeps the start-scan button's height and
        # accent colors, but spans the full width of its row.
        self.confirm_button = QPushButton("开始清理" if not self.english else "Start Cleanup")
        self.confirm_button.setEnabled(False)
        self.confirm_button.clicked.connect(self._clean)
        self.confirm_button.setFixedHeight(30)
        self.confirm_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.confirm_button.setStyleSheet(
            "QPushButton { background: #3A4E78; border: 1px solid #6A84B8; border-radius: 5px; "
            "padding: 4px 12px; color: #FFFFFF; } "
            "QPushButton:hover { background: #45597F; border-color: #7FA3E0; color: #FFFFFF; } "
            "QPushButton:pressed { background: #2E436E; border-color: #6A84B8; } "
            "QPushButton:disabled { color: #737780; border-color: #3A3D42; background: #303236; }"
        )
        buttons.addWidget(self.confirm_button)
        cleanup_card.addLayout(buttons)

        # The log starts empty; it only fills in when cleanup actually runs.
        # Its pane matches the scan-result pane's background/border exactly.
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            "QPlainTextEdit { background: #2A2C31; color: #B8C7E6; border: 1px solid #3E424A; "
            "border-radius: 5px; padding: 6px; font-family: Consolas, 'Courier New', monospace; "
            "font-size: 12px; }"
        )
        self.log_view.setFixedHeight(100)
        cleanup_card.addWidget(self.log_view)

        set_confirm_button(self.scan_button)  # Enter triggers scan first

        # Section cards hug their content: a fixed vertical policy keeps the
        # dialog layout from stretching any card when the window grows.
        for frame in self.findChildren(QFrame):
            if frame.objectName() == "sectionCard":
                frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

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

    def _start_scan(self) -> None:
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

        self._scanning = True
        self.scan_button.setEnabled(False)
        self.confirm_button.setEnabled(False)
        self.result_label.setPlainText("[扫描] …" if not self.english else "[Scan] …")
        threading.Thread(target=self._scan_worker, args=(detected,), daemon=True).start()

    def _on_source_changed(self) -> None:
        """A new dataset directory was chosen (typed or browsed): old scan
        results no longer apply, and the scan button must be usable again."""
        self._scanning = False
        self.scan_button.setEnabled(True)
        self.confirm_button.setEnabled(False)
        self._to_delete_images = []
        self._to_delete_annotations = []
        self._scan_total = self._scan_useful = 0
        self.result_label.setPlainText(
            "路径已更改，请重新开始扫描。" if not self.english
            else "Directory changed; start a new scan."
        )
        self.log_view.clear()

    def _image_annotation_state(self, format_name: str, image: Path, detected) -> str:
        """Fast format-aware check: no PIL decode, no full annotation parse.

        Returns "has" (annotated), "empty" (no usable annotation), or
        "error" (annotation file exists but cannot be read/parsed). Only
        "empty" images are cleanup candidates; "error" files are reported
        separately so the annotator can inspect them.
        """
        try:
            relative = image.relative_to(detected.image_dir)
        except ValueError:
            return "error"
        if format_name == "yolo":
            label_file = detected.annotation_dir / relative.with_suffix(".txt")
            if not label_file.is_file():
                return "empty"
            try:
                return "has" if any(line.strip() for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines()) else "empty"
            except OSError:
                return "error"
        if format_name == "voc":
            xml_file = detected.annotation_dir / relative.with_suffix(".xml")
            if not xml_file.is_file():
                return "empty"
            try:
                import xml.etree.ElementTree as ET
                root = ET.parse(xml_file).getroot()
                for obj in root.iter("object"):
                    bbox = obj.find("bndbox")
                    if bbox is not None and bbox.find("xmin") is not None:
                        return "has"
                return "empty"
            except (OSError, ET.ParseError):
                return "error"
        if format_name == "coco":
            # One JSON read serves every image: any annotation row for the
            # image's basename counts as labeled. The SQLite working copy may
            # not exist yet (dataset never opened in the main window), so
            # fall back to annotations.json in that case.
            if getattr(self, "_coco_annotated_names", None) is None:
                import json as _json
                try:
                    store = CocoAnnotationStore(detected.annotation_dir)
                    document = store.read_document()
                    if not document.get("images"):
                        json_path = AnnotationService._coco_json_path(detected.annotation_dir)
                        if json_path and json_path.is_file():
                            document = _json.loads(json_path.read_text(encoding="utf-8"))
                    image_by_id = {int(item.get("id")): item for item in document.get("images", [])}
                    annotated_ids = {int(ann.get("image_id")) for ann in document.get("annotations", [])}
                    self._coco_annotated_names = {
                        Path(str(image_by_id[image_id].get("file_name", ""))).name
                        for image_id in image_by_id
                        if image_id in annotated_ids
                    }
                except (OSError, ValueError):
                    self._coco_annotated_names = set()
            return "has" if image.name in self._coco_annotated_names else "empty"
        return "error"

    def _scan_worker(self, detected) -> None:
        """Background scan; emits progress and a finished payload.

        Any unexpected failure still emits a finished payload so the UI
        re-enables the scan button (the dialog never dead-ends disabled).
        """
        format_name = detected.format_name
        self._coco_annotated_names = None  # cache built lazily on the first image
        try:
            self._scan_worker_inner(format_name, detected)
        except Exception as exc:
            self.scan_finished.emit({"error": str(exc), "total": 0})

    def _scan_worker_inner(self, format_name: str, detected) -> None:
        from src.services.image_service import SUPPORTED_IMAGE_EXTENSIONS
        images = sorted(path for path in detected.image_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
        useless: list[Path] = []
        problematic: list[Path] = []
        self.scan_progress.emit(0, len(images), 0)
        for index, image in enumerate(images, start=1):
            if index % 5 == 0 or index == len(images):
                self.scan_progress.emit(index, len(images), len(useless))
            state = self._image_annotation_state(format_name, image, detected)
            if state == "empty":
                useless.append(image)
            elif state == "error":
                problematic.append(image)

        # Annotation files whose image no longer exists (keeps folders in
        # sync when images were deleted outside the app). Image names are
        # indexed once so the orphan check is O(1) per file instead of
        # re-walking the whole image tree for every annotation file.
        orphans: list[Path] = []
        if format_name in ("yolo", "voc"):
            image_stems = {path.name.rsplit(".", 1)[0].lower() for path in images}
            suffix = ".txt" if format_name == "yolo" else ".xml"
            for annotation_file in sorted(detected.annotation_dir.rglob(f"*{suffix}")):
                if format_name == "yolo" and annotation_file.name == "classes.txt":
                    continue
                if annotation_file.stem.lower() not in image_stems:
                    orphans.append(annotation_file)

        self.scan_finished.emit({
            "useless": useless,
            "orphans": orphans,
            "problematic": problematic,
            "total": len(images),
        })

    def _on_scan_progress(self, done: int, total: int, useless_count: int) -> None:
        percent = int(done / total * 100) if total else 100
        self.result_label.setPlainText(
            f"[扫描] {done}/{total}  {percent}%"
            if not self.english else
            f"[Scan] {done}/{total}  {percent}%"
        )

    def _summary_line(self, total: int, useful: int, useless: int) -> str:
        """Shared one-line report used by both the scan and cleanup logs."""
        if self.english:
            return f"[Total]: {total}  [Annotated]: {useful}  [Unannotated]: {useless}"
        return f"[总图片]：{total}张  [有效标注]：{useful}张  [无标注]：{useless}张"

    def _on_scan_finished(self, payload: object) -> None:
        self._scanning = False
        self.scan_button.setEnabled(True)
        try:
            self._apply_scan_report(payload)
        except Exception as exc:
            # Never leave the dialog dead-ended: report and re-enable.
            self.confirm_button.setEnabled(False)
            self._to_delete_images = []
            self._to_delete_annotations = []
            self.result_label.setPlainText(
                f"[扫描失败] {exc}\n请重新扫描。"
                if not self.english else
                f"[Scan failed] {exc}\nScan again."
            )

    def _apply_scan_report(self, payload: object) -> None:
        error = str(payload.get("error", "") or "")
        if error:
            self.result_label.setPlainText(
                f"[扫描失败] {error}\n请检查目录后重新扫描。"
                if not self.english else
                f"[Scan failed] {error}\nCheck the directory and scan again."
            )
            self.confirm_button.setEnabled(False)
            self._to_delete_images = []
            self._to_delete_annotations = []
            return
        useless: list[Path] = list(payload.get("useless", []))
        orphans: list[Path] = list(payload.get("orphans", []))
        total: int = int(payload.get("total", 0))
        problematic: list[Path] = list(payload.get("problematic", []))
        self._to_delete_images = useless
        self._to_delete_annotations = orphans
        self._scan_total, self._scan_useful = total, total - len(useless)

        # Summary line first, then one tagged line per file so the report can
        # be checked verbatim against the folder.
        summary = self._summary_line(total, self._scan_useful, len(useless))
        if not (useless or orphans or problematic):
            summary += "  [无需清理]" if not self.english else "  [Nothing to clean]"
        lines = [summary]
        lines += [f"[无标注] {p.name}" for p in useless]
        lines += [f"[孤立标注] {p.name}" for p in orphans]
        lines += [f"[异常] {p.name}" for p in problematic]
        self.result_label.setPlainText("\n".join(lines))
        self.confirm_button.setEnabled(bool(useless or orphans))
        self.scan_button.setDefault(False)
        if useless or orphans:
            set_confirm_button(self.confirm_button)

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
        # The plan lists every file the cleanup will touch, so the progress
        # counter's denominator matches the files the user sees deleted:
        # unannotated images, their mirrored annotation files, then orphans.
        plan: list[Path] = list(self._to_delete_images)
        for path in self._to_delete_images:
            annotation_file = self._annotation_file_for(path)
            if annotation_file is not None and annotation_file.exists():
                plan.append(annotation_file)
        plan += self._to_delete_annotations
        total = len(plan)
        if not AppDialog.question("提示", f"确认清理 {total} 个文件？此操作不可恢复。", self):
            return

        self.confirm_button.setEnabled(False)
        self.log_view.clear()

        def progress(done: int) -> None:
            percent = int(done / total * 100) if total else 100
            self.log_view.setPlainText(
                f"[清理] {done}/{total}  {percent}%"
                if not self.english else
                f"[Clean] {done}/{total}  {percent}%"
            )
            QApplication.processEvents()  # keep the progress line live

        progress(0)
        cleaned: list[Path] = []
        failed: list[Path] = []
        for done, path in enumerate(plan, start=1):
            (cleaned if self._trash(path) else failed).append(path)
            if done % 5 == 0 or done == total:
                progress(done)

        # COCO: drop JSON records for the removed images
        synced_note = ""
        if self._format_name == "coco":
            removed = {path.name for path in self._to_delete_images}
            self._sync_coco_json(removed)
            synced_note = (
                f"[已同步] annotations.json（移除 {len(removed)} 张图片的记录）"
                if not self.english else
                f"[Synced] annotations.json (removed records for {len(removed)} images)"
            )

        # Post-cleanup summary: unannotated images that failed to delete are
        # the only ones left, so they keep the [无标注] count nonzero.
        unannotated = set(self._to_delete_images)
        leftover = sum(1 for path in failed if path in unannotated)
        remaining_total = self._scan_total - (len(self._to_delete_images) - leftover)
        summary = self._summary_line(remaining_total, self._scan_useful, leftover)
        lines = [summary]
        lines += [f"[已清理] {path.name}" for path in cleaned]
        lines += [f"[失败] {path.name}" for path in failed]
        if synced_note:
            lines.append(synced_note)
        lines.append(
            f"[清理完成]  {summary}"
            if not self.english else
            f"[Done]  {summary}"
        )
        self.log_view.setPlainText("\n".join(lines))
        AppDialog.information("提示", f"已清理 {len(cleaned)}/{total} 个文件，详见下方日志。", self)
        self.accept()

    @property
    def cleaned_source(self) -> str:
        return self.source_path.text().strip()
