from __future__ import annotations

from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET
import hashlib
import sqlite3
import uuid
from collections import Counter

from PySide6.QtCore import QEvent, QSettings, QThread, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QIcon, QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QFileDialog, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu, QProgressBar, QSplitter, QVBoxLayout, QWidget

from src.models.annotation import Annotation, LabelPreset, ShapeType, label_color
from src.models.project import ImageRecord, ProjectSettings, ProjectState
from src.services.annotation_service import AnnotationService
from src.services.dataset_detector import DatasetDetector
from src.services.image_service import ImageService
from src.services.dataset_index import DatasetIndexRepository, IndexedImage
from src.services.label_group_store import LabelGroupStore
from src.services.project_service import ProjectService
from src.services.operation_coordinator import OperationCoordinator
from src.services.dataset_session import DatasetScanResult, DatasetSession
from src.services.workers import AutoLabelWorker, DatasetAnnotationSaveWorker, DatasetCountWorker, DatasetScanWorker, DatasetStatisticsWorker, SaveWorker, SingleImageAnnotationWorker
from src.services.format_capabilities import CAPABILITIES, task_for_format
from src.services.yolo_metadata import yolo_keypoint_names
from .annotation_edit_dialog import AnnotationEditDialog
from .common_dialogs import AppDialog
from .conversion_dialog import ConversionDialog, ConversionWorker
from .crosshair_dialog import CrosshairDialog
from .help_dialogs import AboutDialog, FormatGuideDialog, ShortcutsDialog
from .history_dialog import HistoryDialog
from .image_filter_dialog import ImageFilterDialog
from .image_list_panel import ImageListPanel
from .label_groups_dialog import LabelGroupsDialog
from .operations_panel import OperationsPanel
from .preset_panel import PresetPanel
from .settings_dialog import SettingsDialog
from .stats_dialog import StatsDialog
from .task_list_dialog import TaskListDialog
from .task_manager import TaskManager
from .theme import idea_stylesheet
from .support_dialogs import FileInfoDialog, FileTextDialog, ModelStatusBar


DEFAULT_LABEL_NAMES = (
    "person", "head", "hand", "foot", "leg", "knee", "clothes", "coat", "shirt",
    "pants", "dress", "cap", "hat", "glasses", "bag", "shoe", "sneaker", "boot",
    "car", "bus", "truck", "chair", "sofa", "bed", "desk", "lamp", "mouse",
    "phone", "bottle", "vase", "clock", "mirror", "window",
)


def default_label_presets() -> list[LabelPreset]:
    from PySide6.QtGui import QColor
    return [LabelPreset(name, index, QColor.fromHsv((index * 47) % 360, 210, 245).name()) for index, name in enumerate(DEFAULT_LABEL_NAMES)]


class MainWindow(QMainWindow):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ModelLabeling - Annotation Workbench")
        self.resize(1440, 900)
        icon = Path(__file__).resolve().parents[2] / "icon.png"
        if icon.exists(): self.setWindowIcon(QIcon(str(icon)))
        self.annotation_service = AnnotationService()
        self.image_service = ImageService(self.annotation_service)
        self.project_service = ProjectService(self.annotation_service)
        self.settings = ProjectSettings(label_presets=default_label_presets())
        self.label_group_store = LabelGroupStore()
        self.settings.label_groups = self.label_group_store.load_or_initialize(self.settings.label_groups)
        self.state = ProjectState(self.settings)
        self.project_file: Path | None = None
        self.dirty = False
        self.history_store = QSettings("RelinRan", "ModelLabeling")
        self.task_manager = TaskManager(self)
        self.dataset_task_id = self.conversion_task_id = self.auto_task_id = None
        self._dataset_thread = self._conversion_thread = self._auto_thread = None
        self._dataset_worker = self._conversion_worker = self._auto_worker = None
        self._count_thread = self._count_worker = None
        self._annotation_thread = self._annotation_worker = None
        self._annotation_request_path: Path | None = None
        self._stats_thread = self._stats_worker = None
        self._dataset_statistics: dict | None = None
        self._statistics_completed = False
        self._dataset_cancel_requested = False
        self._dataset_load_succeeded = False
        self._dataset_scan_completed = False
        self._dataset_task_dialog_shown = False
        self._dataset_initial_item_selected = False
        self._dataset_dialog_guard = QTimer(self)
        self._dataset_dialog_guard.setInterval(40)
        self._dataset_dialog_guard.timeout.connect(self._hide_unwanted_dataset_dialogs)
        self.dataset_root: Path | None = None
        self.dataset_session: DatasetSession | None = None
        self.dataset_total_images = 0
        self.dataset_indexed_images = 0
        self.dataset_session_id = uuid.uuid4().hex
        self.dataset_index_repository: DatasetIndexRepository | None = None
        self.dataset_current_index = -1
        self.dataset_parser_presets: list[LabelPreset] = []
        self._conversion_cancel_requested = False
        self._auto_cancel_requested = False
        self._save_thread = self._save_worker = None
        self._save_generation = 0
        self._save_generation_in_flight = 0
        self._saved_annotation_labels: dict[str, Counter[str]] = {}
        self._pending_save_statistics: tuple[str, Counter[str], Counter[str]] | None = None
        self._restart_statistics_after_finish = False
        self._auto_save_timer = QTimer(self); self._auto_save_timer.setSingleShot(True); self._auto_save_timer.timeout.connect(self.save_current)
        self.operation_coordinator = OperationCoordinator()
        self._build_ui(); self._apply_annotation_capabilities(); self.task_manager.changed.connect(self._refresh_task_status); self._build_menu(); self._apply_style(); self._update_window_title()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_W and not event.isAutoRepeat():
            self._toggle_canvas_drawing()
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _format_display_name(self) -> str:
        return {
            "yolo_detection": "YOLO Detection",
            "yolo_segmentation": "YOLO Segmentation",
            "yolo_pose": "YOLO Pose",
            "coco": "COCO",
            "voc": "Pascal VOC",
        }.get(self.settings.dataset_task or self.settings.annotation_format, self.settings.annotation_format)

    def _update_window_title(self) -> None:
        root = self.settings.image_dir
        if root and root.name in {"images", "JPEGImages"}:
            root = root.parent
        suffix = f" - {root}" if root else ""
        self.setWindowTitle(f"ModelLabeling - Annotation Workbench{suffix}")

    def _build_ui(self) -> None:
        self.image_panel = ImageListPanel()
        self.canvas = __import__("src.widgets.canvas_view", fromlist=["CanvasView"]).CanvasView()
        self.preset_panel = PresetPanel()
        self.operations_panel = OperationsPanel()
        self.navigation_toolbar = self.operations_panel.navigation_toolbar
        self.preset_panel.set_groups(self.settings.label_groups, self.settings.label_groups[0].name)
        self.image_panel.imageSelected.connect(self.select_image)
        self.image_panel.contextMenuRequested.connect(self._show_image_context_menu)
        self.image_panel.recordsFetched.connect(self._image_records_fetched)
        self.image_panel.list.previousRequested.connect(self.previous_image)
        self.image_panel.list.nextRequested.connect(self.next_image)
        self.image_panel.filtersChanged.connect(self.refresh_image_list)
        self.canvas.dirtyChanged.connect(self._set_dirty)
        self.canvas.annotationSelected.connect(self._canvas_annotation_selected)
        self.canvas.annotationEditRequested.connect(self._edit_annotation)
        self.preset_panel.presetSelected.connect(self._preset_selected)
        self.preset_panel.groupsChanged.connect(self._persist_label_groups)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        # Keep a usable drag target when either sidebar is collapsed to zero.
        self.main_splitter.setHandleWidth(8)
        for widget in (self.image_panel, self.canvas, self.preset_panel): self.main_splitter.addWidget(widget)
        self.main_splitter.setCollapsible(0, True); self.main_splitter.setCollapsible(2, True)
        self.main_splitter.setStretchFactor(1, 1); self.main_splitter.setSizes([335, 785, 320])
        central = QWidget(); central.setObjectName("workbenchCentral"); central.setAutoFillBackground(True); layout = QVBoxLayout(central); layout.setContentsMargins(1, 1, 1, 0); layout.setSpacing(0); layout.addWidget(self.main_splitter); self.setCentralWidget(central)
        self.status_bar = ModelStatusBar(); self.status = QLabel()
        self.status_progress = QProgressBar(); self.status_progress.setObjectName("statusProgress"); self.status_progress.setFixedHeight(8); self.status_progress.setTextVisible(False)
        self.status_progress_current = QLabel("0/...")
        self.status_progress_percent = QLabel("0%")
        progress_layout = QHBoxLayout(); progress_layout.setContentsMargins(0, 0, 0, 0); progress_layout.setSpacing(8)
        progress_layout.addWidget(self.status_progress_current); progress_layout.addWidget(self.status_progress, 1); progress_layout.addWidget(self.status_progress_percent)
        self.status_progress_host = QWidget(); self.status_progress_host.setLayout(progress_layout); self.status_progress_host.setMinimumWidth(190)
        self.status_format = QLabel("格式: YOLO")
        self.status_selected_label = QLabel("已选标签: -")
        self.status_file = QLabel("文件: -")
        self.status_position = QLabel("图片: -/-")
        self.status_size = QLabel("大小: -")
        self.status_resolution = QLabel("分辨率: -")
        self.status_current_count = QLabel("图标签: 0")
        self.status_labeled_count = QLabel("总标注: 0")
        self.status_progress_text = QLabel("总进度: 0%")
        self.status_labeled_count.hide()
        self.status_progress_text.hide()
        self.status_progress_host.hide()
        self.status_task = QLabel("")
        self.status_task.setObjectName("statusTask")
        self.status_saved = QLabel("标注已保存")
        self.status_task.hide()
        self.status.hide()
        status_layout = QHBoxLayout(); status_layout.setContentsMargins(4, 0, 4, 0)
        status_layout.setSpacing(0)
        groups = (
            (self.status_format,),
            (self.status_position, self.status_file, self.status_size, self.status_resolution),
        )
        self.status_selected_label.hide()
        self.status_current_count.hide()
        for group_index, group in enumerate(groups):
            for item in group: status_layout.addWidget(item)
            if group_index < len(groups) - 1:
                separator = QFrame(); separator.setFrameShape(QFrame.Shape.VLine); separator.setFrameShadow(QFrame.Shadow.Plain); separator.setObjectName("statusSeparator"); status_layout.addWidget(separator)
        status_layout.addStretch(1); status_layout.addWidget(self.status_progress_host); status_layout.addWidget(self.status_task); status_layout.addWidget(self.status_saved); status_layout.addWidget(self.status)
        self.status_saved.hide()
        self.status_bar.set_content_layout(status_layout); self.setStatusBar(self.status_bar)
        self._apply_language()

    def _build_menu(self) -> None:
        bar = self.menuBar(); bar.clear()
        english = self.settings.language == "en_US"
        def tr(zh: str, en: str) -> str: return en if english else zh
        file_menu = bar.addMenu("File" if english else "文件"); edit_menu = bar.addMenu("Edit" if english else "编辑"); view_menu = bar.addMenu("View" if english else "查看"); tools_menu = bar.addMenu("Tools" if english else "工具"); help_menu = bar.addMenu("Help" if english else "帮助")
        self.history_menu = file_menu.addMenu(tr("历史  Ctrl+H", "History  Ctrl+H")); self.history_menu_action = self.history_menu.menuAction()
        self.history_menu.aboutToShow.connect(self._refresh_history_menu)
        self._action(self.history_menu, tr("管理历史", "Manage History"), self.open_history)
        self.save_action = self._action(file_menu, tr("保存  Ctrl+S", "Save  Ctrl+S"), self.save_current)
        self._action(file_menu, tr("打开  Ctrl+O", "Open  Ctrl+O"), self.open_directory)
        self._action(file_menu, tr("退出  Ctrl+Q", "Exit  Ctrl+Q"), self.close)
        self._action(edit_menu, tr("标签分组  Ctrl+L+G", "Label Groups  Ctrl+L+G"), self.open_label_groups)
        self._action(edit_menu, tr("应用设置  Ctrl+A+S", "Application Settings  Ctrl+A+S"), self.open_settings)
        self._action(edit_menu, tr("文件筛选  Ctrl+I+F", "File Filter  Ctrl+I+F"), self.open_image_filter)
        self._action(edit_menu, tr("标注辅助  Ctrl+C+A", "Annotation Assist  Ctrl+C+A"), self.open_crosshair)
        self._action(view_menu, tr("适应画布  Ctrl+0", "Fit Canvas  Ctrl+0"), self.canvas.fit_image)
        self._action(view_menu, tr("图片放大  Ctrl++", "Image Zoom In  Ctrl++"), self.canvas.zoom_in)
        self._action(view_menu, tr("图片缩小  Ctrl+-", "Image Zoom Out  Ctrl+-"), self.canvas.zoom_out)
        self._action(view_menu, tr("上张图片  A/↑", "Previous Image  A/↑"), self.previous_image)
        self._action(view_menu, tr("下张图片  D/↓", "Next Image  D/↓"), self.next_image)
        self._action(tools_menu, tr("数据统计  Ctrl+D+S", "Statistics  Ctrl+D+S"), self.open_statistics)
        self._action(tools_menu, tr("数据转换  Ctrl+D+C", "Dataset Conversion  Ctrl+D+C"), self.open_conversion)
        self._action(tools_menu, tr("自动标注  Ctrl+A+L", "Auto Label  Ctrl+A+L"), self.auto_label_all)
        self._action(help_menu, tr("快捷按键", "Shortcuts"), lambda: ShortcutsDialog(self).exec())
        self._action(help_menu, tr("格式说明", "Format Guide"), lambda: FormatGuideDialog(self).exec())
        self._action(help_menu, tr("关于软件", "About Software"), lambda: AboutDialog(self).exec())
        self._shortcuts = []
        shortcut_bindings = (
            ("Ctrl+H", self.open_history),
            ("Ctrl+O", self.open_directory),
            ("Ctrl+S", self.save_current),
            ("Ctrl+Q", self.close),
            # Register both forms for multi-key commands. Qt treats the
            # comma form as a two-stroke sequence; the second form also
            # matches users who keep Ctrl held while pressing the final key.
            ("Ctrl+L, G", self.open_label_groups),
            ("Ctrl+L, Ctrl+G", self.open_label_groups),
            ("Ctrl+A, S", self.open_settings),
            ("Ctrl+A, Ctrl+S", self.open_settings),
            ("Ctrl+I, F", self.open_image_filter),
            ("Ctrl+I, Ctrl+F", self.open_image_filter),
            ("Ctrl+C, A", self.open_crosshair),
            ("Ctrl+C, Ctrl+A", self.open_crosshair),
            ("Ctrl+0", self.canvas.fit_image),
            ("Ctrl++", self.canvas.zoom_in),
            ("Ctrl+-", self.canvas.zoom_out),
            ("A", self.previous_image),
            ("Up", self.previous_image),
            ("D", self.next_image),
            ("Down", self.next_image),
            ("Ctrl+D, S", self.open_statistics),
            ("Ctrl+D, Ctrl+S", self.open_statistics),
            ("Ctrl+D, C", self.open_conversion),
            ("Ctrl+D, Ctrl+C", self.open_conversion),
            ("Ctrl+A, L", self.auto_label_all),
            ("Ctrl+A, Ctrl+L", self.auto_label_all),
            ("W", self._toggle_canvas_drawing),
            ("Escape", self.canvas._disable_draw_mode),
            ("Delete", self.canvas.delete_selected),
            ("Backspace", self.canvas.delete_selected),
        )
        for key, handler in shortcut_bindings:
            shortcut = QShortcut(QKeySequence(key), self); shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut); shortcut.activated.connect(handler); self._shortcuts.append(shortcut)

    def _toggle_canvas_drawing(self) -> None:
        """Toggle drawing without allowing ordinary clicks to create boxes."""
        if self.canvas.draw_enabled:
            self.canvas._disable_draw_mode()
            return
        configured = next(
            (ShapeType(value) for value in self.settings.enabled_shapes if ShapeType(value) in self.canvas.enabled_shapes),
            self.canvas.mode,
        )
        self.canvas.set_mode(configured)
        self.canvas._enable_draw_mode()

    def _enable_canvas_drawing(self) -> None:
        """Backward-compatible entry point for callers that explicitly enable drawing."""
        if not self.canvas.draw_enabled:
            self._toggle_canvas_drawing()

    @staticmethod
    def _action(menu, text, handler):
        action = menu.addAction(text); action.triggered.connect(handler); return action

    def _refresh_history_menu(self) -> None:
        self.history_menu.clear()
        paths = self._history_paths()
        for path in paths[:10]: self._action(self.history_menu, Path(path).name or path, lambda checked=False, path=path: self._start_open_path(Path(path)))
        self.history_menu.addSeparator(); self._action(self.history_menu, "管理历史", self.open_history)

    def _history_paths(self) -> list[str]:
        value = self.history_store.value("history/paths", [])
        return list(value) if isinstance(value, (list, tuple)) else ([str(value)] if value else [])

    def _dataset_location_key(self, root: Path | None = None) -> str:
        location = str((root or self.dataset_root or Path()).resolve())
        return hashlib.sha1(location.casefold().encode("utf-8")).hexdigest()

    def _remember_current_image(self, image_path: Path) -> None:
        if self.dataset_root:
            self.history_store.setValue(f"datasets/current/{self._dataset_location_key()}", str(Path(image_path).resolve()))
            self.history_store.sync()

    def _remembered_current_image(self) -> Path | None:
        if not self.dataset_root:
            return None
        value = self.history_store.value(f"datasets/current/{self._dataset_location_key()}", "")
        return Path(str(value)) if value else None

    def _remember_history(self, path: Path) -> None:
        paths = [str(path)] + [item for item in self._history_paths() if item != str(path)]
        self.history_store.setValue("history/paths", paths[:20]); self.history_store.sync()

    def open_history(self) -> None:
        dialog = HistoryDialog(self._history_paths(), self)
        dialog.historyChanged.connect(lambda paths: (self.history_store.setValue("history/paths", paths), self.history_store.sync()))
        dialog.exec()

    def _apply_style(self) -> None:
        self.setStyleSheet(idea_stylesheet() + "QListView#imageFileList, QListView#imageFileList:focus { background: #25272A; border: 1px solid #464A50; border-radius: 5px; padding: 5px; outline: 0; } QListView#imageFileList::item { height: 30px; padding: 0 5px; margin: 0 15px 2px 0; background: #35383D; color: #FFFFFF; border: 2px solid transparent; border-radius: 5px; } QListView#imageFileList::item:hover { background: #41454C; color: #FFFFFF; border: 2px solid #FFFFFF; } QListView#imageFileList::item:selected, QListView#imageFileList::item:selected:focus { background: #2e436e; color: #FFFFFF; font-weight: 600; border: 2px solid #FFFFFF; outline: 0; } QListWidget#settingsCategories::item:selected { background: #2e436e; } QListWidget#settingsCategories::item:selected:focus { color: #FFFFFF; font-weight: 600; border: none; outline: 0; } QProgressBar#statusProgress { border-radius: 4px; } QProgressBar#statusProgress::chunk { background: #2e436e; border-radius: 4px; }")

    def _apply_language(self) -> None:
        language = self.settings.language
        self.image_panel.set_language(language); self.operations_panel.set_language(language); self.preset_panel.set_language(language)
        if getattr(self, "task_list_dialog", None):
            self.task_list_dialog.set_language(language)
        self.status_saved.setText("Saved" if language == "en_US" and not self.dirty else "Annotation saved" if language == "en_US" else "已保存" if not self.dirty else "未保存")
        self.status_task.setText(self.status_task.text())
        self.refresh_stats()

    def _apply_annotation_capabilities(self) -> None:
        task = task_for_format(self.settings.annotation_format, self.settings.dataset_task)
        supported = CAPABILITIES[task].shapes
        self.canvas.set_enabled_shapes(supported)
        # Shape selection lives in Application Settings now that the legacy
        # shape toolbar is gone. Keep the canvas mode synchronized with it.
        configured = next(
            (ShapeType(value) for value in self.settings.enabled_shapes if ShapeType(value) in supported),
            None,
        )
        self.canvas.set_mode(configured or next(iter(supported), ShapeType.RECTANGLE))
        keypoint_names = []
        if task.value == "yolo_pose" and self.settings.annotation_dir:
            try:
                keypoint_names = yolo_keypoint_names(self.settings.annotation_dir)
            except (OSError, ValueError):
                keypoint_names = []
        self.canvas.set_keypoint_schema(keypoint_names)

    def _apply_live_settings(self, settings: ProjectSettings) -> None:
        self.settings = settings; self.state.settings = settings; self._apply_annotation_capabilities(); self._apply_language(); self._build_menu(); self._apply_style(); self._update_window_title()
        # Return keyboard focus to the work surface after the settings dialog
        # closes. Otherwise an editor/search field can consume W as text and
        # the draw shortcut never reaches the annotation workflow.
        self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)

    def _set_dirty(self, value: bool) -> None:
        self.dirty = value
        if value and self.settings.auto_save and (self.project_file or (self.settings.image_dir and self.settings.annotation_dir)):
            self._save_generation += 1
            self._auto_save_timer.start(300)

    def _refresh_toolbar(self) -> None:
        self.save_action.setVisible(True)
        self.save_action.setEnabled(True)

    @staticmethod
    def _record_from_index(item: IndexedImage) -> ImageRecord:
        return ImageRecord(
            path=item.path, width=item.width, height=item.height,
            file_format=item.path.suffix.lstrip(".").upper(), file_size=item.file_size,
            annotations=[], status="pending", metadata_loaded=False,
        )

    def refresh_image_list(self) -> None:
        if self.dataset_index_repository is not None and self.dataset_total_images:
            query = self.image_panel.search.text()
            status = self.image_panel.selected_status()
            label = self.image_panel.selected_label()
            loader = lambda offset, limit, query=query, status=status, label=label: [
                self._record_from_index(item)
                for item in self.dataset_index_repository.get_page(offset, limit, query, status, label)
            ]
            page = loader(0, 500)
            self.state.images = page
            filtered_total = self.dataset_index_repository.count(query, status, label)
            self.image_panel.set_paged_records(page, filtered_total, loader)
            return
        self.image_panel.set_records(self.image_service.filter_records(self.state.images, self.image_panel.search.text(), self.image_panel.selected_status(), self.image_panel.selected_label()))

    def _annotation_path_for_record(self, record: ImageRecord) -> Path | None:
        annotation_dir = self.settings.annotation_dir
        if not annotation_dir:
            return None
        annotation_dir = Path(annotation_dir)
        if self.settings.annotation_format == "coco":
            if annotation_dir.is_file():
                return annotation_dir if annotation_dir.exists() else None
            return AnnotationService._coco_json_path(annotation_dir)
        suffix = ".xml" if self.settings.annotation_format == "voc" else ".txt"
        try:
            relative_parent = record.path.parent.relative_to(Path(self.settings.image_dir))
        except (ValueError, TypeError):
            relative_parent = Path()
        candidates = (
            annotation_dir / relative_parent / f"{record.path.stem}{suffix}",
            annotation_dir / f"{record.path.stem}{suffix}",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        matches = list(annotation_dir.rglob(f"{record.path.stem}{suffix}")) if annotation_dir.is_dir() else []
        return matches[0] if matches else None

    def _show_image_context_menu(self, record: ImageRecord) -> None:
        annotation_path = self._annotation_path_for_record(record)
        english = self.settings.language == "en_US"
        menu = QMenu(self)
        copy_name_action = menu.addAction("Copy Name" if english else "复制名称")
        annotation_action = None
        if annotation_path is not None:
            annotation_action = menu.addAction("Annotation File" if english else "标注文件")
        info_action = menu.addAction("File Properties" if english else "文件属性")
        chosen = menu.exec(QCursor.pos())
        if chosen is annotation_action and annotation_path is not None:
            try:
                content = self._format_annotation_content(annotation_path)
            except OSError as exc:
                AppDialog.information("Annotation File" if english else "标注文件", str(exc), self)
                return
            FileTextDialog(annotation_path.name, content, self).exec()
        elif chosen is info_action:
            self._show_file_properties(record, annotation_path)
        elif chosen is copy_name_action:
            QApplication.clipboard().setText(record.path.name)

    @staticmethod
    def _format_annotation_content(annotation_path: Path) -> str:
        suffix = annotation_path.suffix.lower()
        raw = annotation_path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".json":
            try:
                return json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                return raw
        if suffix == ".xml":
            try:
                root = ET.fromstring(raw)
                ET.indent(root, space="  ")
                return ET.tostring(root, encoding="unicode")
            except ET.ParseError:
                return raw
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        return "\n".join(f"{index:04d}    {line}" for index, line in enumerate(lines, 1)) or "(empty)"

    def _show_file_properties(self, record: ImageRecord, annotation_path: Path | None) -> None:
        english = self.settings.language == "en_US"
        image_size = record.file_size
        units = ("B", "KB", "MB", "GB")
        unit = 0
        while image_size >= 1024 and unit < len(units) - 1:
            image_size /= 1024
            unit += 1
        annotation_size = "-"
        if annotation_path is not None:
            try:
                value = annotation_path.stat().st_size
                annotation_size = f"{value / 1024:.1f} KB" if value >= 1024 else f"{value} B"
            except OSError:
                pass
        rows = (
            [
                ("Image Path", str(record.path)),
                ("Image Format", record.file_format),
                ("Image Size", f"{image_size:.1f} {units[unit]}"),
                ("Image Dimensions", f"{record.width} x {record.height}"),
                ("Annotation Path", str(annotation_path or "-")),
                ("Annotation Size", annotation_size),
            ] if english else [
                ("图片路径", str(record.path)),
                ("图片格式", record.file_format),
                ("图片大小", f"{image_size:.1f} {units[unit]}"),
                ("图片尺寸", f"{record.width} x {record.height}"),
                ("标注路径", str(annotation_path or "-")),
                ("标注大小", annotation_size),
            ]
        )
        FileInfoDialog("File Properties" if english else "文件属性", rows, self).exec()

    def select_image(self, row: int) -> None:
        records = self.image_panel.records
        if not 0 <= row < len(records): return
        record = records[row]; self.state.current_index = row; self.dataset_current_index = self.dataset_index_repository.position(record.path) if self.dataset_index_repository else row; self._remember_current_image(record.path); self.canvas.load_image(QImage(str(record.path)), record.annotations); self.image_panel.select_record(record); self.canvas.set_image_info(record.path.name, self.dataset_current_index + 1, self.dataset_total_images or len(self.state.images), record.file_format, record.file_size); self.refresh_stats(); self._load_selected_annotations(record)
        if self.dataset_session is not None:
            self.dataset_session.current_path = record.path

    def _image_records_fetched(self, records: list[ImageRecord]) -> None:
        """Keep the main state in sync with rows loaded by the paged list."""
        if self.dataset_index_repository is None or not records:
            return
        known = {str(item.path) for item in self.state.images}
        self.state.images.extend(item for item in records if str(item.path) not in known)

    def previous_image(self) -> None:
        self._navigate_image_list(-1)

    def next_image(self) -> None:
        self._navigate_image_list(1)

    def _navigate_image_list(self, step: int) -> None:
        """Navigate exactly like clicking the adjacent visible image item."""
        records = self.image_panel.records
        if not records:
            return
        current = self.state.current_image
        row = next((index for index, record in enumerate(records) if record.path == getattr(current, "path", None)), -1)
        if row < 0:
            row = 0 if step > 0 else len(records) - 1
        if step > 0 and row >= len(records) - 1 and self.image_panel.list.model().canFetchMore():
            self.image_panel.list.model().fetchMore()
            records = self.image_panel.records
        target_row = max(0, min(len(records) - 1, row + step))
        self.image_panel.list.setCurrentRow(target_row)

    def _select_state_image(self) -> None:
        record = self.state.current_image
        if record is None: return
        self.dataset_current_index = self.dataset_index_repository.position(record.path) if self.dataset_index_repository else self.state.current_index
        self._remember_current_image(record.path); self.refresh_image_list(); self.canvas.load_image(QImage(str(record.path)), record.annotations); self.image_panel.select_record(record); self.canvas.set_image_info(record.path.name, self.dataset_current_index + 1, self.dataset_total_images or len(self.state.images), record.file_format, record.file_size); self.refresh_stats(); self._load_selected_annotations(record)

    def _load_selected_annotations(self, record: ImageRecord) -> None:
        if self._annotation_thread is not None:
            self._annotation_request_path = record.path
            return
        if record.metadata_loaded and record.status != "pending":
            return
        if not self.settings.image_dir or not self.settings.annotation_dir:
            return
        self._annotation_request_path = record.path
        self._annotation_thread = QThread(self)
        self._annotation_worker = SingleImageAnnotationWorker(
            record, self.settings.image_dir, self.settings.annotation_dir,
            ProjectSettings.from_dict(self.settings.to_dict()),
        )
        self._annotation_worker.settings.label_presets = list(self.dataset_parser_presets or self.settings.label_presets)
        self._annotation_worker.moveToThread(self._annotation_thread)
        self._annotation_thread.started.connect(self._annotation_worker.run)
        self._annotation_worker.finished.connect(self._selected_annotations_loaded)
        self._annotation_worker.finished.connect(self._annotation_thread.quit)
        self._annotation_thread.finished.connect(self._annotation_thread_finished)
        self._annotation_thread.start()

    def _selected_annotations_loaded(self, path: str, width: int, height: int, file_format: str, annotations: list, error: str) -> None:
        record = next((item for item in self.state.images if str(item.path) == path), None)
        if record is None:
            return
        record.width, record.height, record.file_format = width, height, file_format
        record.annotations, record.error = annotations, error or None
        record.status = "error" if error else ("labeled" if annotations else "unlabeled")
        record.metadata_loaded = True
        self._saved_annotation_labels[path] = Counter(annotation.label for annotation in annotations)
        if self.state.current_image is record:
            self.canvas.load_image(QImage(str(record.path)), record.annotations)
            self.canvas.set_image_info(record.path.name, self.dataset_current_index + 1, self.dataset_total_images or len(self.state.images), record.file_format, record.file_size)
            self.refresh_stats()
        self.image_panel.update_record(record)

    def _annotation_thread_finished(self) -> None:
        thread = self._annotation_thread
        self._annotation_thread = self._annotation_worker = None
        if thread is not None:
            thread.deleteLater()
        pending_path = self._annotation_request_path
        self._annotation_request_path = None
        if pending_path is not None:
            record = next((item for item in self.state.images if item.path == pending_path), None)
            if record is not None and record is self.state.current_image:
                QTimer.singleShot(0, lambda record=record: self._load_selected_annotations(record))

    def refresh_stats(self) -> None:
        stats = self.state.statistics(); current = self.state.current_image
        english = self.settings.language == "en_US"
        format_name = self._format_display_name()
        self.status_format.setText(f"Format: {format_name}" if english else f"格式: {format_name}")
        if current:
            size = current.file_size
            units = ("B", "KB", "MB", "GB")
            unit = 0
            while size >= 1024 and unit < len(units) - 1: size /= 1024; unit += 1
            self.status_file.setText(f"File: {current.path.name}" if english else f"文件: {current.path.name}")
            total_images = self.dataset_total_images if self.dataset_total_images > 0 else "..."
            current_position = self.dataset_current_index + 1 if self.dataset_current_index >= 0 else self.state.current_index + 1
            self.status_position.setText(f"Image: {current_position}/{total_images}" if english else f"图片: {current_position}/{total_images}")
            self.status_size.setText(f"Size: {size:.1f} {units[unit]}" if english else f"大小: {size:.1f} {units[unit]}")
            self.status_resolution.setText(f"Resolution: {current.width}x{current.height}" if english else f"分辨率: {current.width}x{current.height}")
        else:
            self.status_position.setText("Image: -/-" if english else "图片: -/-"); self.status_file.setText("File: -" if english else "文件: -"); self.status_size.setText("Size: -" if english else "大小: -"); self.status_resolution.setText("Resolution: -" if english else "分辨率: -")
        self.status_saved.setText("Unsaved" if english and self.dirty else "Saved" if english else "未保存" if self.dirty else "已保存")
        self._refresh_task_status()

    def _refresh_task_status(self) -> None:
        tasks = self.task_manager.tasks()
        if not tasks:
            self.status_task.setText("")
            return
        english = self.settings.language == "en_US"
        parts = []
        for task in tasks:
            count = f"{task.current}/{task.total}" if task.total else ""
            parts.append(f"{task.name} {count} {task.progress}%".replace("  ", " ").strip())
        self.status_task.setText(("Tasks: " if english else "任务: ") + " | ".join(parts))

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            self._apply_live_settings(dialog.settings)
            QTimer.singleShot(0, self._close_confirm_residue)

    def open_label_groups(self) -> None:
        dialog = LabelGroupsDialog(self.settings.label_groups, self, self.settings.language)
        if dialog.exec() == dialog.DialogCode.Accepted:
            try:
                self.label_group_store.save_groups(dialog.groups)
            except (OSError, sqlite3.Error) as exc:
                AppDialog.information("标签分组保存失败", str(exc), self)
                return
            self.settings.label_groups = dialog.groups
            selected = self.preset_panel.selected_group().name if self.preset_panel.selected_group() else None
            self.preset_panel.set_groups(self.settings.label_groups, selected)
            QTimer.singleShot(0, self._close_confirm_residue)

    def _persist_label_groups(self) -> None:
        """Persist edits made directly in the main label panel."""
        try:
            self.label_group_store.save_groups(self.preset_panel.groups)
            self.settings.label_groups = self.preset_panel.groups
        except (OSError, sqlite3.Error) as exc:
            AppDialog.information("标签分组保存失败", str(exc), self)

    def open_image_filter(self) -> None:
        dialog = ImageFilterDialog(self.image_panel.search.text(), self.image_panel.selected_status(), self.image_panel.selected_label(), self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            query, status, label = dialog.values(); self.image_panel.search.setText(query); self.image_panel.status.setCurrentIndex(max(0, self.image_panel.status.findData(status))); self.image_panel.set_label_filter(label); self.refresh_image_list(); QTimer.singleShot(0, self._close_confirm_residue)

    def open_crosshair(self) -> None:
        dialog = CrosshairDialog(self.settings.crosshair_line_width, self.settings.crosshair_color, self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            self.settings.crosshair_line_width, self.settings.crosshair_color = dialog.line_width.value(), dialog._color.name(); self.canvas.set_crosshair_settings(self.settings.crosshair_line_width, self.settings.crosshair_color); QTimer.singleShot(0, self._close_confirm_residue)

    def open_statistics(self) -> None:
        # The progress bar is updated before the worker thread fully exits.
        # Use the scan completion signal as the source of truth so a dataset
        # that has finished indexing is not blocked by Qt thread teardown.
        dataset_loading = (
            self._dataset_thread is not None
            and self._dataset_thread.isRunning()
            and not self._dataset_scan_completed
        )
        if dataset_loading:
            english = self.settings.language == "en_US"
            AppDialog.information(
                "Statistics" if english else "数据统计",
                "The dataset is still loading. Please try again after loading finishes." if english else "数据集仍在加载，请等待加载完成后再查看数据统计。",
                self,
            )
            return
        if not self.state.images: AppDialog.information("数据统计", "请先打开数据集", self); return
        if self._stats_thread is not None and self._stats_thread.isRunning():
            english = self.settings.language == "en_US"
            AppDialog.information(
                "Statistics" if english else "数据统计",
                "Statistics are still being calculated. Please try again later." if english else "数据集还在统计中，请等待统计完再查看",
                self,
            )
            return
        StatsDialog(self._dataset_statistics or self.state.statistics(), self.settings.language, self).exec()

    def open_conversion(self) -> None:
        if self._operation_blocked("conversion"):
            return
        dialog = ConversionDialog(self.settings.label_presets, self)
        if dialog.exec() == dialog.DialogCode.Accepted and dialog.options: self._start_conversion(dialog.options)

    def _operation_blocked(self, operation: str) -> bool:
        coordinator = self.operation_coordinator
        coordinator.dataset_loading = self._dataset_thread is not None and self._dataset_thread.isRunning()
        coordinator.auto_labeling = self._auto_thread is not None and self._auto_thread.isRunning()
        coordinator.converting = self._conversion_thread is not None and self._conversion_thread.isRunning()
        coordinator.statistics_running = self._stats_thread is not None and self._stats_thread.isRunning()
        coordinator.statistics_complete = self._statistics_completed
        allowed, active = coordinator.can_start(operation)
        if not allowed and active != "statistics":
            english = self.settings.language == "en_US"
            AppDialog.information(
                "Busy" if english else "任务执行中",
                f"{active} is running. Please wait until it finishes." if english else f"{active}正在执行，请等待完成后再操作",
                self,
            )
            return True
        if not allowed and active == "statistics":
            english = self.settings.language == "en_US"
            AppDialog.information(
                "Statistics" if english else "数据统计",
                "Statistics are still running. Please wait until statistics finish." if english else "数据集还在统计中，请等待统计完再操作",
                self,
            )
            return True
        return False

    def open_task_list(self) -> None:
        self._show_task_list()

    def _show_task_list(self) -> None:
        if self._dataset_thread is not None and self._dataset_thread.isRunning():
            self._dataset_task_dialog_shown = True
        if not getattr(self, "task_list_dialog", None):
            self.task_list_dialog = TaskListDialog(self.task_manager, self, self.settings.language); self.task_list_dialog.taskStopped.connect(self._task_stopped)
        dialog = self.task_list_dialog
        dialog.show(); dialog.refresh(); dialog.raise_(); dialog.activateWindow()

    def _close_task_list(self) -> None:
        dialog = getattr(self, "task_list_dialog", None)
        if dialog is not None:
            # Remove the task window completely before any completion message
            # is created. Closing alone can leave a native top-level surface
            # visible for one frame on Windows.
            dialog.hide()
            dialog.close()
            dialog.deleteLater()
            self.task_list_dialog = None

    def _close_confirm_residue(self) -> None:
        """Remove stray top-level windows left behind by a completed dialog."""
        task_dialog = getattr(self, "task_list_dialog", None)
        for widget in QApplication.topLevelWidgets():
            if widget is self or widget is task_dialog or not widget.isVisible():
                continue
            if isinstance(widget, QMenu):
                continue
            # Confirmation has already returned to the main window. Any
            # remaining top-level window here is an unintended residue.
            widget.close()

    def _task_stopped(self, name: str) -> None:
        AppDialog.information(name, f"停止{name}成功", self)

    def open_directory(self) -> None:
        root = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if root: self._start_open_path(Path(root))

    def _start_open_path(self, root: Path) -> None:
        if self._operation_blocked("dataset"):
            return
        if self._save_thread is not None and self._save_thread.isRunning():
            AppDialog.information("保存进行中", "请等待当前标注保存完成后再切换数据集", self)
            return
        self._export_coco_checkpoint()
        requested_root = Path(root).resolve()
        if self.dataset_root and requested_root == self.dataset_root.resolve():
            language = self.settings.language
            AppDialog.information(
                "Dataset Already Open" if language == "en_US" else "数据集已打开",
                "The current dataset is already open." if language == "en_US" else "当前数据集已经打开。",
                self,
            )
            return
        try:
            detected = DatasetDetector.detect(requested_root)
        except ValueError as exc:
            AppDialog.information("打开数据集失败", str(exc), self); return
        self.dataset_session = DatasetSession.from_detected(detected)
        self.dataset_root = self.dataset_session.root; self.settings.annotation_format = self.dataset_session.format_name; self.settings.dataset_task = self.dataset_session.task_name; self.settings.image_dir = self.dataset_session.image_dir; self.settings.annotation_dir = self.dataset_session.annotation_dir; self._apply_annotation_capabilities(); self._update_window_title(); self.refresh_stats(); self._remember_history(root); self._start_dataset_scan(self.dataset_session.image_dir, self.dataset_session.annotation_dir)

    def _start_dataset_scan(self, image_dir: Path, annotation_dir: Path) -> None:
        self._stop_dataset_workers_for_switch()
        self.dataset_total_images = 0
        self.dataset_indexed_images = 0
        self.dataset_parser_presets = []
        self.dataset_current_index = -1
        self.dataset_index_repository = DatasetIndexRepository(
            image_dir.parent.resolve(), image_dir, annotation_dir, self.settings.annotation_format
        )
        self.state.images = []
        self.state.current_index = -1
        self.image_panel.set_records([])
        self._dataset_statistics = None
        self._statistics_completed = False
        self._saved_annotation_labels.clear()
        self._pending_save_statistics = None
        self._restart_statistics_after_finish = False
        if self._stats_thread is not None and self._stats_thread.isRunning():
            self._stats_worker.cancelled = True
            for signal, slot in (
                (self._stats_worker.progress, self._dataset_statistics_progress),
                (self._stats_worker.finished, self._dataset_statistics_finished),
                (self._stats_worker.failed, self._dataset_statistics_failed),
            ):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
            try:
                self._stats_thread.finished.disconnect(self._stats_thread_finished)
            except (RuntimeError, TypeError):
                pass
            self._stats_thread.quit()
            self._stats_thread.wait()
            self._stats_thread = self._stats_worker = None
        self._dataset_cancel_requested = False; self._dataset_load_succeeded = False; self._dataset_scan_completed = False; self._dataset_task_dialog_shown = False; self._dataset_initial_item_selected = False; self.dataset_root = image_dir.parent.resolve(); self._set_dataset_loading(True, "加载"); self._dataset_dialog_guard.start(); self._dataset_thread = QThread(self); self._dataset_worker = DatasetScanWorker(image_dir, annotation_dir, ProjectSettings.from_dict(self.settings.to_dict()), self.dataset_root, self.dataset_session_id); self.dataset_task_id = self.task_manager.start("打开数据集", self.cancel_dataset_scan, 0); self._dataset_worker.moveToThread(self._dataset_thread); self._dataset_thread.started.connect(self._dataset_worker.run); self._dataset_worker.progress.connect(self._dataset_scan_progress); self._dataset_worker.partial.connect(self._dataset_scan_partial); self._dataset_worker.finished.connect(self._dataset_scan_finished); self._dataset_worker.failed.connect(self._dataset_scan_failed); self._dataset_worker.finished.connect(self._dataset_thread.quit); self._dataset_worker.failed.connect(self._dataset_thread.quit); self._dataset_thread.finished.connect(self._dataset_thread_finished); QTimer.singleShot(0, self._dataset_thread.start)
        self._count_thread = QThread(self)
        self._count_worker = DatasetCountWorker(image_dir, self.dataset_session_id)
        self._count_worker.moveToThread(self._count_thread)
        self._count_thread.started.connect(self._count_worker.run)
        self._count_worker.finished.connect(self._dataset_count_ready)
        self._count_worker.finished.connect(self._count_thread.quit)
        self._count_thread.finished.connect(self._count_thread_finished)
        self._count_thread.start()


    def _stop_dataset_workers_for_switch(self) -> None:
        """Finish old dataset workers before replacing their QObject refs."""
        count_thread = self._count_thread
        if count_thread is not None and count_thread.isRunning():
            if self._count_worker is not None:
                self._count_worker.cancelled = True
            count_thread.quit()
            count_thread.wait()
        self._count_thread = self._count_worker = None
        dataset_thread = self._dataset_thread
        if dataset_thread is not None and dataset_thread.isRunning():
            self._dataset_cancel_requested = True
            if self._dataset_worker is not None:
                self._dataset_worker.cancelled = True
                for signal, slot in (
                    (self._dataset_worker.progress, self._dataset_scan_progress),
                    (self._dataset_worker.partial, self._dataset_scan_partial),
                    (self._dataset_worker.finished, self._dataset_scan_finished),
                    (self._dataset_worker.failed, self._dataset_scan_failed),
                ):
                    try:
                        signal.disconnect(slot)
                    except (RuntimeError, TypeError):
                        pass
            try:
                dataset_thread.finished.disconnect(self._dataset_thread_finished)
            except (RuntimeError, TypeError):
                pass
            dataset_thread.quit()
            dataset_thread.wait()
        if self.dataset_task_id is not None:
            self.task_manager.finish(self.dataset_task_id)
        self._dataset_thread = self._dataset_worker = None

        annotation_thread = self._annotation_thread
        if annotation_thread is not None and annotation_thread.isRunning():
            if self._annotation_worker is not None and hasattr(self._annotation_worker, "cancelled"):
                self._annotation_worker.cancelled = True
            annotation_thread.quit()
            annotation_thread.wait()
        self._annotation_thread = self._annotation_worker = None
        self._annotation_request_path = None

    def _show_dataset_task_if_running(self) -> None:
        if self._dataset_thread is not None and self._dataset_thread.isRunning() and self.dataset_task_id is not None:
            self._dataset_task_dialog_shown = True
            self._show_task_list()

    def _hide_unwanted_dataset_dialogs(self) -> None:
        task_dialog = getattr(self, "task_list_dialog", None)
        for widget in QApplication.topLevelWidgets():
            if widget is self or widget is task_dialog or isinstance(widget, (QMenu, AppDialog)) or not isinstance(widget, QWidget) or not widget.isVisible():
                continue
            # A real application dialog has a title and a useful content size.
            # The stray native surface has neither and must never cover the canvas.
            if not widget.windowTitle().strip() or widget.width() < 240 or widget.height() < 160:
                widget.hide()
                widget.close()

    def _dataset_scan_progress(self, current: int, total: int) -> None:
        self.dataset_indexed_images = current
        if total:
            percent = int(current / total * 100)
            if current >= total:
                self._show_statistics_transition()
            else:
                self._set_status_progress("加载", current, total, percent)
        else:
            self._set_status_progress("加载", current, 0)
        self.task_manager.update(self.dataset_task_id, percent if total else 0, current, total)

    def _dataset_count_ready(self, total: int, session_id: str) -> None:
        if session_id != self.dataset_session_id or total <= 0:
            return
        self.dataset_total_images = total
        if self.dataset_session is not None:
            self.dataset_session.total_images = total
        if self._dataset_worker is not None and self._dataset_worker.session_id == session_id:
            self._dataset_worker.total_count = total
        percent = int(self.dataset_indexed_images / total * 100) if total else 0
        if self.dataset_indexed_images >= total:
            self._show_statistics_transition()
        else:
            self._set_status_progress("加载", self.dataset_indexed_images, total, percent)
        self.image_panel.set_total_count(total)
        self.refresh_stats()

    def _count_thread_finished(self) -> None:
        thread = self._count_thread
        self._count_thread = self._count_worker = None
        if thread is not None:
            thread.deleteLater()

    def _dataset_scan_partial(self, result: DatasetScanResult) -> None:
        """Show indexed images while the worker continues parsing annotations."""
        if result.session_id and result.session_id != self.dataset_session_id:
            return
        if self._dataset_cancel_requested:
            return
        records = result.records
        if result.total_images:
            self.dataset_total_images = result.total_images
        if result.presets:
            self.dataset_parser_presets = list(result.presets)
        current_path = self.state.current_image.path if self.state.current_image else None
        should_select_initial = not self._dataset_initial_item_selected
        if result.append_only:
            self.state.images.extend(records)
            records = self.state.images
        else:
            records = result.records
        restored_index = 0 if records and not self._dataset_initial_item_selected else next(
            (index for index, record in enumerate(records) if current_path and record.path == current_path),
            0 if records else -1,
        )
        self._dataset_initial_item_selected = bool(records)
        self.state.images, self.state.current_index = records, restored_index
        if result.append_only:
            self.image_panel.append_records(result.records)
        else:
            self.refresh_image_list()
        # The first batch is delivered from a worker thread. Force the model,
        # splitter layout, and canvas viewport to repaint immediately so the
        # user can start annotating as soon as the batch arrives.
        self.image_panel.list.updateGeometry()
        self.image_panel.list.viewport().update()
        self.canvas.updateGeometry()
        self.canvas.viewport().update()
        self.centralWidget().layout().activate()
        if records and should_select_initial:
            self._select_state_image()

    def _dataset_scan_finished(self, records) -> None:
        if isinstance(records, DatasetScanResult) and records.session_id and records.session_id != self.dataset_session_id:
            return
        if self._dataset_worker and self._dataset_worker.cancelled: self._set_dataset_loading(False); self._close_task_list(); return
        self._show_statistics_transition()
        self._dataset_scan_completed = True
        append_only_result = False
        if isinstance(records, DatasetScanResult):
            scan_result = records
            if scan_result.total_images:
                self.dataset_total_images = scan_result.total_images
            if scan_result.append_only:
                append_only_result = True
                records = self.state.images
            else:
                records = scan_result.records
            if scan_result.presets:
                self.dataset_parser_presets = list(scan_result.presets)
        current_path = self.state.current_image.path if self.state.current_image else None
        restored_index = next(
            (index for index, record in enumerate(records) if current_path and record.path == current_path),
            0 if records else -1,
        )
        self.state.images, self.state.current_index = records, restored_index
        # Switch the status bar before clearing the loading state so there is
        # no hidden-frame gap between "加载 100%" and "统计 0%".
        self._start_dataset_statistics()
        self._set_dataset_loading(False)
        self._dataset_load_succeeded = True
        session_id = self.dataset_session_id
        QTimer.singleShot(
            0,
            lambda: self._finish_dataset_scan_ui(records, append_only_result, session_id),
        )

    def _finish_dataset_scan_ui(self, records, append_only_result: bool, session_id: str) -> None:
        """Refresh the heavy image view after the progress status has painted."""
        if session_id != self.dataset_session_id or self._dataset_cancel_requested:
            return
        # Rebuild the page through the current search/status filters.  The
        # scan completion callback can run after the filter dialog, and must
        # not restore an unfiltered page over the user's selection.
        self.refresh_image_list()
        self.refresh_stats()
        if records and not append_only_result:
            self._select_state_image()

    def _dataset_scan_failed(self, message: str) -> None:
        self._dataset_scan_completed = False
        self._dataset_dialog_guard.stop(); self._set_dataset_loading(False); self._close_task_list()
        if self.isVisible(): AppDialog.information("打开数据集失败", message, self)

    def _start_dataset_statistics(self) -> None:
        if self._stats_thread is not None or not self.settings.image_dir or not self.settings.annotation_dir:
            return
        self._statistics_completed = False
        self._set_status_progress("统计", 0, self.dataset_total_images)
        self._stats_thread = QThread(self)
        self._stats_worker = DatasetStatisticsWorker(
            None, self.settings.image_dir, self.settings.annotation_dir,
            ProjectSettings.from_dict(self.settings.to_dict()), self.dataset_parser_presets,
        )
        self._stats_worker.moveToThread(self._stats_thread)
        self._stats_thread.started.connect(self._stats_worker.run)
        self._stats_worker.progress.connect(self._dataset_statistics_progress)
        self._stats_worker.finished.connect(self._dataset_statistics_finished)
        self._stats_worker.failed.connect(self._dataset_statistics_failed)
        self._stats_worker.finished.connect(self._stats_thread.quit)
        self._stats_worker.failed.connect(self._stats_thread.quit)
        self._stats_thread.finished.connect(self._stats_thread_finished)
        self._stats_thread.start()

    def _dataset_statistics_progress(self, current: int, total: int, snapshot: dict) -> None:
        self._dataset_statistics = snapshot
        percent = int(current / total * 100) if total else 0
        self._set_status_progress("统计", current, total, percent)

    def _dataset_statistics_finished(self, snapshot: dict) -> None:
        self._dataset_statistics = snapshot
        self._statistics_completed = True
        total = int(snapshot.get("total_images", 0)) if snapshot else 0
        self._set_status_progress("统计", total, total, 100 if total else 0)

    def _dataset_statistics_failed(self, message: str) -> None:
        self._dataset_statistics = None
        self._statistics_completed = True

    def _stats_thread_finished(self) -> None:
        thread = self._stats_thread
        self._stats_thread = self._stats_worker = None
        if thread is not None:
            thread.deleteLater()
        if self._restart_statistics_after_finish:
            self._restart_statistics_after_finish = False
            self._start_dataset_statistics()
            return
        if self._dataset_scan_completed and self._auto_thread is None and self._conversion_thread is None:
            self.status_progress_host.setVisible(False)

    def _dataset_thread_finished(self) -> None:
        self.task_manager.finish(self.dataset_task_id); self.dataset_task_id = None; self._dataset_thread = self._dataset_worker = None
        self._dataset_dialog_guard.stop()
        self._close_task_list()
        self._dataset_task_dialog_shown = False

    def cancel_dataset_scan(self) -> None:
        self._dataset_cancel_requested = True
        if self._dataset_worker: self._dataset_worker.cancelled = True
        if self._count_worker: self._count_worker.cancelled = True

    def _set_status_progress(self, kind: str, current: int, total: int, percent: int | None = None) -> None:
        english = self.settings.language == "en_US"
        label = {"加载": "Loading", "统计": "Statistics", "转换": "Conversion", "自动标注": "Auto labeling"}.get(kind, kind) if english else kind
        self.status_progress_host.setVisible(True)
        if total:
            value = max(0, min(100, int(percent if percent is not None else current / total * 100)))
            self.status_progress.setRange(0, 100); self.status_progress.setValue(value)
            self.status_progress_current.setText(f"{label} {current}/{total}")
            self.status_progress_percent.setText(f"{value}%")
        else:
            self.status_progress.setRange(0, 0)
            self.status_progress_current.setText(f"{label} {current}/...")
            self.status_progress_percent.setText("--")

    def _show_statistics_transition(self) -> None:
        """Replace a completed loading bar while the statistics task starts."""
        english = self.settings.language == "en_US"
        self.status_progress_host.setVisible(True)
        self.status_progress.setRange(0, 0)
        self.status_progress_current.setText(
            "Preparing statistics" if english else "正在准备统计"
        )
        self.status_progress_percent.setText("--")

    def _set_dataset_loading(self, active: bool, kind: str = "加载") -> None:
        # Indexing is non-blocking: the first batch is usable immediately.
        for widget in (self.canvas, self.image_panel): widget.setEnabled(True)
        if active:
            self.status_progress_host.setVisible(True)
            self._set_status_progress(kind, 0, 0)
        else:
            if self._stats_thread is not None:
                self.status_progress_host.setVisible(True)
                return
            self.status_progress.setRange(0, 100)
            self.status_progress.setValue(100)
            label = "Loading" if self.settings.language == "en_US" else "加载"
            self.status_progress_current.setText(f"{label} {self.dataset_total_images}/{self.dataset_total_images}" if self.dataset_total_images else f"{label} 0/0")
            self.status_progress_percent.setText("100%" if self.dataset_total_images else "0%")
            self.status_progress_host.setVisible(False)
        self.preset_panel.setEnabled(True)

    @staticmethod
    def _count_images(directory: Path) -> int:
        return sum(1 for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in ImageService.__dict__.get("SUPPORTED_IMAGE_EXTENSIONS", {".jpg", ".jpeg", ".png", ".bmp", ".webp"}))

    def save_current(self) -> None:
        current = self.state.current_image
        if current is None: self.dirty = False; return
        if self._save_thread is not None and self._save_thread.isRunning():
            return
        self._save_generation_in_flight = self._save_generation
        save_settings = ProjectSettings.from_dict(self.settings.to_dict())
        save_settings.label_presets = list(self.dataset_parser_presets or self.settings.label_presets)
        path_key = str(current.path)
        new_labels = Counter(annotation.label for annotation in self.canvas.annotations)
        old_labels = self._saved_annotation_labels.get(path_key, Counter(new_labels))
        self._pending_save_statistics = (path_key, Counter(old_labels), Counter(new_labels))
        self._save_thread = QThread(self)
        if self.project_file:
            self._save_worker = SaveWorker(self.project_file, current.path, list(self.canvas.annotations), save_settings)
        elif self.settings.image_dir and self.settings.annotation_dir:
            self._save_worker = DatasetAnnotationSaveWorker(current.path, self.settings.image_dir, self.settings.annotation_dir, list(self.canvas.annotations), save_settings)
        else:
            self.dirty = False
            return
        self._save_worker.moveToThread(self._save_thread); self._save_thread.started.connect(self._save_worker.run); self._save_worker.finished.connect(self._save_finished); self._save_worker.finished.connect(self._save_thread.quit); self._save_thread.finished.connect(self._save_thread_finished); self._save_thread.start()

    def _save_thread_finished(self) -> None:
        thread = self._save_thread
        self._save_thread = None
        self._save_worker = None
        if thread is not None: thread.deleteLater()

    def _save_finished(self, error: str) -> None:
        if not error:
            self._apply_saved_annotation_statistics()
        else:
            self._pending_save_statistics = None
        if error: AppDialog.information("保存失败", error, self)
        elif self._save_generation == self._save_generation_in_flight:
            self.dirty = False
        elif self.settings.auto_save:
            self._auto_save_timer.start(50)
        if not error and not self.settings.auto_save:
            self._export_coco_checkpoint()

    def _export_coco_checkpoint(self) -> None:
        if self.settings.annotation_format != "coco" or not self.settings.annotation_dir:
            return
        try:
            AnnotationService.export_coco(self.settings.annotation_dir)
        except (OSError, ValueError) as exc:
            if self.isVisible():
                AppDialog.information("COCO 导出失败", str(exc), self)

    def _apply_saved_annotation_statistics(self) -> None:
        pending = self._pending_save_statistics
        self._pending_save_statistics = None
        if pending is None:
            return
        path_key, old_labels, new_labels = pending
        self._saved_annotation_labels[path_key] = Counter(new_labels)
        if self.dataset_index_repository is not None:
            image_path = Path(path_key)
            annotation_path = None
            if self.settings.annotation_dir:
                if self.settings.annotation_format == "coco":
                    annotation_path = self.settings.annotation_dir / ".model_labeling.sqlite3"
                elif self.settings.image_dir:
                    try:
                        relative = image_path.relative_to(self.settings.image_dir)
                        suffix = ".xml" if self.settings.annotation_format == "voc" else ".txt"
                        annotation_path = self.settings.annotation_dir / relative.with_suffix(suffix)
                    except ValueError:
                        annotation_path = None
            self.dataset_index_repository.update_annotation(image_path, annotation_path, new_labels.elements())
        if self._stats_thread is not None and self._stats_thread.isRunning():
            self._restart_statistics_after_finish = True
            return
        if self._dataset_statistics is None:
            return
        statistics = dict(self._dataset_statistics)
        counts = Counter(statistics.get("label_counts", {}))
        counts.subtract(old_labels)
        counts.update(new_labels)
        counts = Counter({label: count for label, count in counts.items() if count > 0})
        total = int(statistics.get("total_images", self.dataset_total_images))
        statistics["total_labels"] = max(
            0,
            int(statistics.get("total_labels", 0))
            - sum(old_labels.values())
            + sum(new_labels.values()),
        )
        statistics["labeled_images"] = max(
            0,
            int(statistics.get("labeled_images", 0)) - int(bool(old_labels)) + int(bool(new_labels)),
        )
        statistics["percentage"] = statistics["labeled_images"] / total * 100.0 if total else 0.0
        statistics["label_counts"] = dict(sorted(counts.items()))
        self._dataset_statistics = statistics

    def _start_conversion(self, options) -> None:
        self._conversion_cancel_requested = False; self._conversion_thread = QThread(self); self._conversion_worker = ConversionWorker(options); self.conversion_task_id = self.task_manager.start("数据集转换", self.cancel_conversion, self._count_images(options.source_path)); self._conversion_worker.moveToThread(self._conversion_thread); self._conversion_thread.started.connect(self._conversion_worker.run); self._conversion_worker.progress.connect(self._conversion_progress); self._conversion_worker.completed.connect(self._conversion_finished); self._conversion_worker.failed.connect(self._conversion_failed); self._conversion_worker.completed.connect(self._conversion_thread.quit); self._conversion_worker.failed.connect(self._conversion_thread.quit); self._conversion_thread.finished.connect(self._conversion_thread_finished); self._set_status_progress("转换", 0, self.task_manager.tasks()[-1].total if self.task_manager.tasks() else 0); self._conversion_thread.start()

    def _conversion_progress(self, current: int, total: int) -> None:
        percent = int(current / total * 100) if total else 0
        self.task_manager.update(self.conversion_task_id, percent, current, total)
        self._set_status_progress("转换", current, total, percent)

    def _conversion_finished(self, report) -> None:
        if self._conversion_cancel_requested: return
        self.status_progress_host.setVisible(False)
        if report.failed: self._conversion_failed("; ".join(report.errors[:3]))
        else:
            AppDialog.information("数据集转换", "数据集转换完成", self)
        # Successful background operations finish silently; progress is
        # already visible in the task/status area.

    def _conversion_failed(self, message: str) -> None:
        self.status_progress_host.setVisible(False)
        if not self._conversion_cancel_requested: AppDialog.information("数据集转换失败", message, self)
    def _conversion_thread_finished(self) -> None:
        self.status_progress_host.setVisible(False); self.task_manager.finish(self.conversion_task_id); self.conversion_task_id = None; self._conversion_thread = self._conversion_worker = None; self._maybe_close_task_list()
    def cancel_conversion(self) -> None:
        self._conversion_cancel_requested = True
        if self._conversion_worker: self._conversion_worker.cancelled = True

    def auto_label_all(self) -> None:
        if self._operation_blocked("auto"):
            return
        if not self.state.images: AppDialog.information("自动标注", "请先打开数据集", self); return
        if not self.settings.onnx_model_path: AppDialog.information("自动标注", "请先在应用设置中选择 ONNX 模型", self); return
        auto_settings = ProjectSettings.from_dict(self.settings.to_dict())
        auto_settings.label_presets = list(
            self.dataset_parser_presets or self.settings.label_presets
        )
        self._auto_cancel_requested = False
        self._auto_thread = QThread(self)
        records = None if self.dataset_index_repository is not None else self.state.images
        self._auto_worker = AutoLabelWorker(records, auto_settings)
        self.auto_task_id = self.task_manager.start(
            "自动标注", self.cancel_auto_label, len(self.state.images),
        )
        self._auto_worker.moveToThread(self._auto_thread)
        self._auto_thread.started.connect(self._auto_worker.run)
        self._auto_worker.modelReady.connect(self._auto_model_ready)
        self._auto_worker.progress.connect(self._auto_progress)
        self._auto_worker.finished.connect(self._auto_finished)
        self._auto_worker.failed.connect(self._auto_failed)
        self._auto_worker.finished.connect(self._auto_thread.quit)
        self._auto_worker.failed.connect(self._auto_thread.quit)
        self._auto_thread.finished.connect(self._auto_thread_finished)
        self._set_status_progress("自动标注", 0, len(self.state.images))
        self._auto_thread.start()

    def _auto_model_ready(
        self, task: str, keypoint_names: list[str], class_names: list[str],
    ) -> None:
        if self._auto_worker is not None:
            try:
                self._auto_worker.annotationReady.connect(self._auto_annotation_ready, Qt.ConnectionType.UniqueConnection)
            except (TypeError, RuntimeError):
                pass
        if class_names:
            colors = {preset.class_id: preset.color for preset in self.dataset_parser_presets}
            self.dataset_parser_presets = [
                LabelPreset(
                    name, class_id,
                    colors.get(class_id, label_color(name)),
                )
                for class_id, name in enumerate(class_names)
            ]
        if task == "pose" and self.settings.annotation_format in {"yolo", "coco"}:
            if self.settings.annotation_format == "yolo":
                self.settings.dataset_task = "yolo_pose"
            self._apply_annotation_capabilities()
            self.canvas.set_keypoint_schema(keypoint_names)
        elif task in {"segment", "segmentation"} and self.settings.annotation_format in {"yolo", "coco"}:
            if self.settings.annotation_format == "yolo":
                self.settings.dataset_task = "yolo_segmentation"
            self._apply_annotation_capabilities()

    def _auto_annotation_ready(self, path: str, annotations: list[Annotation]) -> None:
        """Commit worker output on the GUI thread instead of mutating records in a worker."""
        image_path = Path(path)
        if self.dataset_index_repository is not None:
            annotation_path = None
            if self.settings.annotation_dir:
                if self.settings.annotation_format == "coco":
                    annotation_path = Path(self.settings.annotation_dir) / ".model_labeling.sqlite3"
                elif self.settings.image_dir:
                    try:
                        relative = image_path.relative_to(self.settings.image_dir)
                        suffix = ".xml" if self.settings.annotation_format == "voc" else ".txt"
                        annotation_path = Path(self.settings.annotation_dir) / relative.with_suffix(suffix)
                    except ValueError:
                        annotation_path = None
            self.dataset_index_repository.update_annotation(
                image_path, annotation_path,
                (annotation.label for annotation in annotations),
            )
        record = next((item for item in self.state.images if str(item.path) == path), None)
        if record is None:
            return
        record.annotations = [Annotation.from_dict(item.to_dict()) for item in annotations]
        record.status = "labeled" if record.annotations else "unlabeled"
        record.metadata_loaded = True
        if self.state.current_image is record:
            self.canvas.load_image(QImage(str(record.path)), record.annotations)
        self.image_panel.update_record(record)

    def _auto_progress(self, current: int, total: int) -> None:
        percent = int(current / total * 100) if total else 0
        self.task_manager.update(self.auto_task_id, percent, current, total)
        self._set_status_progress("自动标注", current, total, percent)

    def _auto_finished(self) -> None:
        if self._auto_cancel_requested: return
        self.status_progress_host.setVisible(False)
        self.refresh_image_list(); self.refresh_stats()
        self._export_coco_checkpoint()
        self._dataset_statistics = None
        self._statistics_completed = False
        self._start_dataset_statistics()
        AppDialog.information("自动标注", "自动标注完成", self)
    def _auto_failed(self, message: str) -> None:
        if self._auto_cancel_requested: return
        self._set_dataset_loading(False); AppDialog.information("自动标注失败", message, self)

    def _auto_label_failed(self, message: str) -> None:
        self.status.setText("")
        self._auto_failed(message)
    def cancel_auto_label(self) -> None:
        self._auto_cancel_requested = True
        if self._auto_worker: self._auto_worker.cancelled = True
    def _auto_thread_finished(self) -> None:
        self.status_progress_host.setVisible(False); self.task_manager.finish(self.auto_task_id); self.auto_task_id = None; self._auto_thread = self._auto_worker = None; self._maybe_close_task_list()
        if self._auto_cancel_requested:
            self.refresh_image_list()
            self._dataset_statistics = None
            self._statistics_completed = False
            self._start_dataset_statistics()

    def _maybe_close_task_list(self) -> None:
        if not self.task_manager.tasks(): self._close_task_list()

    def _stop_background_tasks_for_exit(self) -> None:
        """Cancel every worker before the window and its QObject children die."""
        # Keep this centralized: task-list cancellation and window-close
        # cancellation must use the same worker flags.
        for task in self.task_manager.tasks():
            self.task_manager.cancel(task.task_id)

        if self._stats_worker is not None:
            self._stats_worker.cancelled = True

        threads = (
            self._dataset_thread,
            self._conversion_thread,
            self._auto_thread,
            self._annotation_thread,
            self._stats_thread,
            self._save_thread,
        )
        for thread in threads:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()

    def closeEvent(self, event) -> None:
        self._stop_background_tasks_for_exit()
        self._export_coco_checkpoint()
        self._close_task_list()
        event.accept()

    def _canvas_annotation_selected(self, annotation) -> None:
        # Canvas selection is independent from the label-list selection.
        # Clicking a box must never move the right-panel selection.
        self.preset_panel.clear_selection()
        self.refresh_stats()

    def _preset_selected(self, label: str) -> None:
        color = label_color(label)
        self.canvas.set_current_label(label, color)
        if self.canvas.update_selected_label(label, color):
            self.refresh_stats()
            self.save_current()

    def _edit_annotation(self, annotation) -> None:
        dialog = AnnotationEditDialog(self.settings.label_groups, annotation.label, self, self.settings.language, annotation)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        if dialog.deleted:
            self.canvas.delete_selected()
            self.save_current()
            return
        label = dialog.selected_label()
        color = label_color(label)
        if self.canvas.update_selected_label(label, color):
            # The dialog edits keypoint coordinates/visibility on the same
            # model object. Refresh scene geometry before the autosave starts.
            self.canvas.refresh_annotations()
            self.refresh_stats()
            self.save_current()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.main_splitter.sizes()[0] != 335 or self.main_splitter.sizes()[2] != 320:
            self.main_splitter.setSizes([335, max(1, self.main_splitter.width() - 655), 320])
