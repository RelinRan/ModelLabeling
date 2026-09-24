from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget

from src.models.annotation import LabelPreset
from src.services.annotation_service import AnnotationService
from src.services.coco_store import CocoAnnotationStore
from src.services.cleanup import FixPolicy, fix_dataset, scan_dataset
from src.services.cleanup.report import RULES, Severity
from src.services.cleanup.runner import DEFAULT_MIN_BOX_SIZE as MIN_BOX_SIZE
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
    clean_progress = Signal(str)
    clean_finished = Signal(object)

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
        # Shorter-side threshold for the "tiny box" finding. It is an
        # annotation policy rather than a property of the data, so it is a
        # control instead of a constant, and the report names the value used.
        self.min_box_size = QSpinBox()
        self.min_box_size.setRange(1, 999)
        self.min_box_size.setValue(MIN_BOX_SIZE)
        self.min_box_size.setSuffix(" px")
        self.min_box_size.setFixedHeight(30)
        source_form.addRow("最小框尺寸" if not self.english else "Min box size", self.min_box_size)
        source_card.addLayout(source_form)

        # ---- scan + report ---------------------------------------------------
        self.scan_button = QPushButton("开始扫描" if not self.english else "Start Scan")
        self.scan_button.clicked.connect(self._start_scan)
        self.scan_button.setFixedHeight(30)
        source_card.addWidget(self.scan_button)
        self.scan_progress.connect(self._on_scan_progress)
        self.scan_finished.connect(self._on_scan_finished)
        self.clean_progress.connect(self._on_clean_progress)
        self.clean_finished.connect(self._on_clean_finished)
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

        # The two switches the whole pass is about. Structural repairs have
        # one defensible answer each, so they are on by default; a quality
        # rule encodes annotation policy -- how small is too small, whether a
        # box-less image is a negative sample -- so it is the user's call.
        policy_box = QVBoxLayout()
        policy_box.setSpacing(2)
        self.fix_structural = QCheckBox(
            "结构修复（标注与图片不一致，自动修正）" if not self.english
            else "Structural repairs (fix annotation/image mismatches)"
        )
        self.fix_structural.setChecked(True)
        self.fix_structural.toggled.connect(self._update_policy_detail)
        self.fix_quality = QCheckBox(
            "质量策略（删除过小/重复框、无标注图片）" if not self.english
            else "Quality policy (drop tiny/duplicate boxes, unannotated images)"
        )
        self.fix_quality.setChecked(False)
        self.fix_quality.toggled.connect(self._update_policy_detail)
        policy_box.addWidget(self.fix_structural)
        policy_box.addWidget(self.fix_quality)
        self.policy_detail = QLabel()
        self.policy_detail.setWordWrap(True)
        self.policy_detail.setStyleSheet(
            "QLabel { background: transparent; border: none; color: #8E97A8; "
            "font-size: 11px; }"
        )
        policy_box.addWidget(self.policy_detail)
        cleanup_card.addLayout(policy_box)
        self._update_policy_detail()

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
        # Read the widgets here: the scan runs on a worker thread, which must
        # not touch them.
        self._min_box_size = self.min_box_size.value()

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
            # COCO file_name may contain a relative directory. Basename-only
            # matching attaches annotations to every same-named image in a
            # nested dataset, so keep normalized relative keys instead.
            if getattr(self, "_coco_annotated_keys", None) is None:
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
                    annotated_keys = {
                        self._normalized_relative_name(image_by_id[image_id].get("file_name", ""))
                        for image_id in image_by_id
                        if image_id in annotated_ids
                    }
                    # Legacy flat COCO files sometimes store only a basename.
                    # Permit that fallback only when the basename is unique on
                    # disk; duplicate basenames must remain separate images.
                    disk_name_counts: dict[str, int] = {}
                    for candidate in detected.image_dir.rglob("*"):
                        if candidate.is_file():
                            key = candidate.name.casefold()
                            disk_name_counts[key] = disk_name_counts.get(key, 0) + 1
                    self._coco_annotated_keys = annotated_keys
                    self._coco_annotated_unique_names = {
                        key for key in annotated_keys
                        if "/" not in key and disk_name_counts.get(Path(key).name.casefold(), 0) == 1
                    }
                except (OSError, ValueError):
                    self._coco_annotated_keys = set()
                    self._coco_annotated_unique_names = set()
            relative_key = self._normalized_relative_name(relative.as_posix())
            basename_key = image.name.casefold()
            return "has" if (
                relative_key in self._coco_annotated_keys
                or basename_key in self._coco_annotated_unique_names
            ) else "empty"
        return "error"

    def _scan_worker(self, detected) -> None:
        """Background scan; emits progress and a finished payload.

        Any unexpected failure still emits a finished payload so the UI
        re-enables the scan button (the dialog never dead-ends disabled).
        """
        format_name = detected.format_name
        self._coco_annotated_keys = None  # cache built lazily on the first image
        self._coco_annotated_unique_names = set()
        try:
            self._scan_worker_inner(format_name, detected)
        except Exception as exc:
            self.scan_finished.emit({"error": str(exc), "total": 0})

    def _scan_worker_inner(self, format_name: str, detected) -> None:
        from src.services.image_service import SUPPORTED_IMAGE_EXTENSIONS
        images = sorted(path for path in detected.image_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
        self.scan_progress.emit(0, len(images), 0)

        # The validator decodes every image, which is by far the slower half
        # of a scan, so it drives the live progress line. The deletion pass
        # below only reads text and runs quietly afterwards.
        plan, validation_error = self._validate(detected)
        useless, problematic = self._unusable_images(format_name, images, detected)

        self.scan_finished.emit({
            "useless": useless,
            "orphans": self._orphan_annotations(format_name, images, detected),
            "problematic": problematic,
            "total": len(images),
            "plan": plan,
            "validation_error": validation_error,
        })

    def _validate(self, detected) -> tuple[object | None, str]:
        """Run the read-only validator.

        A validator failure must not cancel the scan: the deletion pass is
        what the cleanup acts on, so its result is still worth reporting. The
        error is surfaced in the report instead of being swallowed.
        """
        try:
            plan = scan_dataset(
                detected.root,
                presets=self.presets,
                detected=detected,
                min_box_size=getattr(self, "_min_box_size", MIN_BOX_SIZE),
                progress=lambda done, count: self.scan_progress.emit(done, count, 0),
            )
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        return plan, ""

    def _unusable_images(self, format_name: str, images: list[Path], detected) -> tuple[list[Path], list[Path]]:
        useless: list[Path] = []
        problematic: list[Path] = []
        for image in images:
            state = self._image_annotation_state(format_name, image, detected)
            if state == "empty":
                useless.append(image)
            elif state == "error":
                problematic.append(image)
        return useless, problematic

    def _orphan_annotations(self, format_name: str, images: list[Path], detected) -> list[Path]:
        """Annotation files whose image no longer exists (keeps folders in
        sync when images were deleted outside the app). Image names are
        indexed once so the orphan check is O(1) per file instead of
        re-walking the whole image tree for every annotation file."""
        if format_name not in ("yolo", "voc"):
            return []
        image_keys = {
            self._normalized_relative_name(path.relative_to(detected.image_dir).with_suffix("").as_posix())
            for path in images
        }
        suffix = ".txt" if format_name == "yolo" else ".xml"
        orphans: list[Path] = []
        for annotation_file in sorted(detected.annotation_dir.rglob(f"*{suffix}")):
            if format_name == "yolo" and annotation_file.name.casefold() == "classes.txt":
                continue
            annotation_key = self._normalized_relative_name(
                annotation_file.relative_to(detected.annotation_dir).with_suffix("").as_posix()
            )
            if annotation_key not in image_keys:
                orphans.append(annotation_file)
        return orphans

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
        findings = self._validation_lines(
            payload.get("plan"), str(payload.get("validation_error", "") or ""),
        )
        summary = self._summary_line(total, self._scan_useful, len(useless))
        # "Nothing to clean" has to mean the whole report, not just the files
        # this dialog would delete: a validator finding is work too.
        if not (useless or orphans or problematic or findings):
            summary += "  [无需清理]" if not self.english else "  [Nothing to clean]"
        lines = [summary]
        lines += [f"[无标注] {p.name}" for p in useless]
        lines += [f"[孤立标注] {p.name}" for p in orphans]
        lines += [f"[异常] {p.name}" for p in problematic]
        lines += findings
        self.result_label.setPlainText("\n".join(lines))
        # A validator finding is work too, and the repair pass rescans, so the
        # button is live whenever the scan reported anything at all. Whether a
        # click does something is the policy's business, not the scan's.
        has_work = bool(useless or orphans or problematic or findings)
        self.confirm_button.setEnabled(has_work)
        self.scan_button.setDefault(False)
        if has_work:
            set_confirm_button(self.confirm_button)

    def _severity_name(self, severity: Severity) -> str:
        if self.english:
            return {
                Severity.STRUCTURAL: "structural",
                Severity.REVIEW: "review",
                Severity.INFO: "info",
            }[severity]
        return {
            Severity.STRUCTURAL: "结构",
            Severity.REVIEW: "待确认",
            Severity.INFO: "提示",
        }[severity]

    def _validation_lines(self, plan, error: str) -> list[str]:
        """Render the validator's findings, grouped by rule.

        Grouped rather than one line per finding: a real dataset produced
        1235 findings for a single rule, which would bury every other line in
        the pane. Each rule shows its count and one example file, so the
        report stays checkable against the folder.
        """
        if error:
            return [
                (f"[校验失败] {error}\n其余检查已跳过，清理仍可进行。"
                 if not self.english else
                 f"[Validation failed] {error}\nRemaining checks were skipped; cleanup still works.")
            ]
        if plan is None or not plan.issues:
            return []

        grouped = plan.by_rule()
        buckets: dict[Severity, list[tuple[str, int]]] = {
            Severity.STRUCTURAL: [], Severity.REVIEW: [], Severity.INFO: [],
        }
        for rule, values in grouped.items():
            buckets[RULES[rule].severity].append((rule, len(values)))

        totals = "  ".join(
            f"{self._severity_name(severity)} {sum(count for _rule, count in buckets[severity])}"
            for severity in (Severity.STRUCTURAL, Severity.REVIEW, Severity.INFO)
        )
        lines = [
            f"[校验] {totals}  [图片] {plan.total_images}"
            if not self.english else
            f"[Validation] {totals}  [Images] {plan.total_images}"
        ]
        for severity in (Severity.STRUCTURAL, Severity.REVIEW, Severity.INFO):
            for rule, count in sorted(buckets[severity], key=lambda item: (-item[1], item[0])):
                tag = RULES[rule].tag[1 if self.english else 0]
                lines.append(f"[{tag}] {count}  ({rule})")
                lines.append(f"    {grouped[rule][0].target.name}")
        return lines

    @staticmethod
    def _normalized_relative_name(value) -> str:
        return str(value or "").replace("\\", "/").removeprefix("./").casefold()

    def _sync_coco_json(self, removed_image_keys: set[str]) -> None:
        """Remove only the exact relative COCO image records deleted on disk."""
        try:
            store = CocoAnnotationStore(self._annotation_dir)
            document = store.read_document()
            if not document.get("images"):
                import json as _json
                json_path = AnnotationService._coco_json_path(self._annotation_dir)
                if json_path and json_path.is_file():
                    document = _json.loads(json_path.read_text(encoding="utf-8"))
            normalized_removed = {self._normalized_relative_name(item) for item in removed_image_keys}
            rows = list(document.get("images", []))
            document_name_counts: dict[str, int] = {}
            for item in rows:
                name = Path(self._normalized_relative_name(item.get("file_name", ""))).name
                document_name_counts[name] = document_name_counts.get(name, 0) + 1

            def is_removed(item) -> bool:
                key = self._normalized_relative_name(item.get("file_name", ""))
                if key in normalized_removed:
                    return True
                # Legacy basename-only rows are safe only when unique.
                return (
                    "/" not in key
                    and document_name_counts.get(Path(key).name, 0) == 1
                    and any(Path(removed).name == Path(key).name for removed in normalized_removed)
                )

            removed_ids = {
                int(item.get("id"))
                for item in rows
                if is_removed(item)
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

    def _policy(self) -> FixPolicy:
        rules: set[str] = set()
        if self.fix_structural.isChecked():
            rules |= FixPolicy.structural().rules
        if self.fix_quality.isChecked():
            rules |= FixPolicy.quality_rules()
        return FixPolicy.of(rules)

    def _update_policy_detail(self) -> None:
        structural = len(FixPolicy.structural().rules)
        quality = len(FixPolicy.quality_rules())
        if self.english:
            text = (f"{structural} structural repairs, {quality} quality rules; "
                    "files removed are backed up and the run is logged")
        else:
            text = (f"结构修复 {structural} 项，质量策略 {quality} 项；"
                    "删除前自动备份，并生成清理清单")
        self.policy_detail.setText(text)

    def _clean(self) -> None:
        """Run the repair pass, then report what it did.

        The deletion this dialog always did is now one rule among several:
        the fixer additionally corrects the declared header of a file, drops
        boxes the annotation policy rejects, and records every removal in a
        backup and a manifest. It rescans between passes, so the cascade the
        rules imply -- the last box goes, the image is now box-less, the image
        goes -- plays out instead of being guessed in a single pass.
        """
        source = Path(self.source_path.text().strip())
        if not source.is_dir():
            return
        policy = self._policy()
        if not policy.rules:
            AppDialog.information("提示", "没有勾选任何修复策略。", self)
            return
        try:
            detected = DatasetDetector.detect(source)
        except ValueError:
            AppDialog.information("提示", "无法识别该数据集格式。", self)
            return

        message = (
            f"按当前策略修复 {source} ？\n\n"
            f"结构修复：{'开' if self.fix_structural.isChecked() else '关'}\n"
            f"质量策略：{'开' if self.fix_quality.isChecked() else '关'}\n\n"
            "删除的文件会先备份，并生成清理清单。"
            if not self.english else
            f"Repair {source} with the current policy?\n\n"
            f"Structural: {'on' if self.fix_structural.isChecked() else 'off'}\n"
            f"Quality: {'on' if self.fix_quality.isChecked() else 'off'}\n\n"
            "Removed files are backed up and the run is logged."
        )
        if not AppDialog.question("提示", message, self):
            return

        self._scanning = True
        self.confirm_button.setEnabled(False)
        self.scan_button.setEnabled(False)
        self.log_view.setPlainText("[修复] …" if not self.english else "[Repair] …")
        self._min_box_size = self.min_box_size.value()
        threading.Thread(
            target=self._clean_worker, args=(detected, policy), daemon=True,
        ).start()

    def _on_clean_progress(self, text: str) -> None:
        self.log_view.setPlainText(f"[修复] {text}" if not self.english else f"[Repair] {text}")

    def _clean_worker(self, detected, policy) -> None:
        """Background repair; always emits, so the dialog never dead-ends."""
        try:
            report = fix_dataset(
                detected.root,
                presets=self.presets,
                policy=policy,
                detected=detected,
                min_box_size=getattr(self, "_min_box_size", MIN_BOX_SIZE),
                progress=lambda text: self.clean_progress.emit(text),
            )
        except Exception as exc:
            self.clean_finished.emit({"error": f"{type(exc).__name__}: {exc}"})
            return
        self.clean_finished.emit({"report": report})

    def _on_clean_finished(self, payload: object) -> None:
        self._scanning = False
        self.scan_button.setEnabled(True)
        self.confirm_button.setEnabled(False)
        if not isinstance(payload, dict) or "report" not in payload:
            message = str(payload.get("error")) if isinstance(payload, dict) else str(payload)
            self.log_view.setPlainText(
                f"[失败] {message}" if not self.english else f"[Failed] {message}"
            )
            AppDialog.information("提示", f"修复失败：{message}", self)
            return

        report = payload["report"]
        removed = report.removed
        rewritten = report.rewritten

        # Recomputed rather than derived from the report: the same fast check
        # the scan uses, so the summary line after a repair means exactly what
        # it meant before one.
        source = Path(self.source_path.text().strip())
        total = useful = useless = 0
        try:
            detected = DatasetDetector.detect(source)
            from src.services.image_service import SUPPORTED_IMAGE_EXTENSIONS
            images = sorted(
                path for path in detected.image_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )
            total = len(images)
            for image in images:
                if self._image_annotation_state(detected.format_name, image, detected) == "empty":
                    useless += 1
                else:
                    useful += 1
        except ValueError:
            pass
        summary = self._summary_line(total, useful, useless)

        counts: dict[str, int] = {}
        for repair in report.repairs:
            counts[repair.rule] = counts.get(repair.rule, 0) + 1

        lines = [summary]
        lines += [f"[已清理] {Path(entry['path']).name}" for entry in removed]
        for rule, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            tag = RULES[rule].tag[1 if self.english else 0]
            lines.append(
                f"[修复] {tag} {count}  ({rule})" if not self.english
                else f"[Repaired] {tag} {count}  ({rule})"
            )
        if rewritten:
            lines.append(
                f"[已改写] {len(rewritten)} 个标注文件（已备份原文件）" if not self.english
                else f"[Rewritten] {len(rewritten)} annotation files (originals backed up)"
            )
        if report.backup_dir is not None:
            lines.append(
                f"[备份] {report.backup_dir}" if not self.english
                else f"[Backup] {report.backup_dir}"
            )
        lines += [f"[失败] {error}" for error in report.errors]
        lines.append(
            f"[清理完成]  {summary}" if not self.english else f"[Done]  {summary}"
        )
        self.log_view.setPlainText("\n".join(lines))
        self.accept()

    @property
    def cleaned_source(self) -> str:
        return self.source_path.text().strip()

