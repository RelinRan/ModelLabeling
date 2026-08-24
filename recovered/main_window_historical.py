from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QThread, Qt, Signal, QObject, QTimer
from PySide6.QtGui import QIcon, QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import QFileDialog, QLabel, QMainWindow, QProgressBar, QSplitter, QStatusBar, QHBoxLayout, QVBoxLayout, QWidget

from src.models.annotation import LabelPreset
from src.models.project import ProjectSettings, ProjectState
from src.services.annotation_service import AnnotationService
from src.services.image_service import ImageService
from src.services.project_service import ProjectService
from .canvas_view import CanvasView
from .common_dialogs import AppDialog
from .help_dialogs import AboutDialog, ShortcutsDialog
from .image_list_panel import ImageListPanel
from .operations_panel import OperationsPanel
from .preset_panel import PresetPanel
from .settings_dialog import SettingsDialog
from .task_list_dialog import TaskListDialog
from .task_manager import TaskManager
from .theme import idea_stylesheet

DEFAULT_LABEL_NAMES = ("person", "head", "hand", "foot", "leg", "knee", "clothes", "coat", "shirt", "pants", "dress", "cap", "hat", "glasses", "bag", "shoe", "sneaker", "boot", "car", "bus", "truck", "chair", "sofa", "bed", "desk", "lamp", "mouse", "phone", "bottle", "vase", "clock", "mirror", "window")

def default_label_presets():
    from PySide6.QtGui import QColor
    return [LabelPreset(name, i, QColor.fromHsv((i * 47) % 360, 210, 245).name()) for i, name in enumerate(DEFAULT_LABEL_NAMES)]

class DatasetScanWorker(QObject):
    progress = Signal(int, int); finished = Signal(object); failed = Signal(str)
    def __init__(self, image_dir, annotation_dir, settings):
        super().__init__(); self.image_dir=image_dir; self.annotation_dir=annotation_dir; self.settings=settings; self.cancelled=False
    def run(self):
        try: self.finished.emit(ImageService(AnnotationService()).scan(self.image_dir, self.annotation_dir, self.settings, self.progress.emit, lambda: self.cancelled))
        except Exception as exc: self.failed.emit(str(exc))

class SaveWorker(QObject):
    finished = Signal()
    def __init__(self, project_path, image_path, annotations, settings):
        super().__init__(); self.args = project_path, image_path, annotations, settings
    def run(self):
        try: ProjectService(AnnotationService()).save_current(*self.args)
        finally: self.finished.emit()


class ModelStatusBar(QStatusBar):
    def set_content_layout(self, layout): self._content_layout = layout
    def layout(self): return getattr(self, "_content_layout", super().layout())

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent); self.setWindowTitle("ModelLabeling - Annotation Workbench"); self.resize(1440,900)
        icon=Path(__file__).resolve().parents[2]/"icon.png"
        if icon.exists(): self.setWindowIcon(QIcon(str(icon)))
        self.annotation_service=AnnotationService(); self.image_service=ImageService(self.annotation_service); self.project_service=ProjectService(self.annotation_service)
        self.settings=ProjectSettings(label_presets=default_label_presets()); self.state=ProjectState(self.settings); self.project_file=None; self.dirty=False
        self.history_store=QSettings("RelinRan","ModelLabeling"); self.task_manager=TaskManager(self); self.dataset_task_id=None; self._dataset_thread=None; self._dataset_worker=None; self._save_thread=None; self._save_worker=None; self._auto_save_timer=QTimer(self); self._auto_save_timer.setSingleShot(True); self._auto_save_timer.timeout.connect(self.save_current)
        self._build_ui(); self._build_menu(); self._apply_style()

    def _build_ui(self):
        self.image_panel=ImageListPanel(); self.canvas=CanvasView(); self.preset_panel=PresetPanel(); self.operations_panel=OperationsPanel()
        self.navigation_toolbar=self.operations_panel.navigation_toolbar; self.preset_panel.set_groups(self.settings.label_groups, self.settings.label_groups[0].name); self.preset_panel.set_presets(self.settings.label_presets)
        self.image_panel.imageSelected.connect(self.select_image); self.image_panel.filtersChanged.connect(self.refresh_image_list); self.canvas.dirtyChanged.connect(self._set_dirty)
        self.main_splitter=QSplitter(Qt.Orientation.Horizontal); [self.main_splitter.addWidget(w) for w in (self.image_panel,self.canvas,self.preset_panel)]; self.main_splitter.setCollapsible(0,True); self.main_splitter.setCollapsible(2,True); self.main_splitter.setStretchFactor(0,0); self.main_splitter.setStretchFactor(1,1); self.main_splitter.setStretchFactor(2,0); self.main_splitter.setSizes([360,720,360])
        central=QWidget(); layout=QVBoxLayout(central); layout.setContentsMargins(0,0,0,0); layout.addWidget(self.main_splitter); self.setCentralWidget(central)
        self.status_bar=ModelStatusBar(); self.status=QLabel(); self.status_progress=QProgressBar(); self.status_progress.setObjectName("statusProgress"); self.status_progress.setFixedHeight(8); self.status_progress_text=QLabel("总进度: 0%"); self.status_current_count=QLabel("图标签: 0"); self.status_labeled_count=QLabel("总标注: 0"); status_layout=QHBoxLayout(); status_layout.setContentsMargins(4,0,4,0); status_layout.addWidget(self.status_progress); status_layout.addWidget(self.status_progress_text); status_layout.addWidget(self.status); self.status_bar.set_content_layout(status_layout); self.setStatusBar(self.status_bar); self._apply_language()

    def _build_menu(self):
        bar=self.menuBar(); bar.clear(); file=bar.addMenu("文件"); edit=bar.addMenu("编辑"); view=bar.addMenu("查看"); help_menu=bar.addMenu("帮助")
        self.history_menu=file.addMenu("历史  Ctrl+H"); self.history_menu_action=self.history_menu.menuAction(); self.history_menu.addAction("管理历史")
        self.save_action=file.addAction("保存  Ctrl+S"); self.save_action.triggered.connect(self.save_current); file.addAction("打开  Ctrl+O",self.open_directory); file.addAction("退出  Ctrl+Q",self.close); edit.addAction("应用设置  Ctrl+A+S",self.open_settings); view.addAction("任务列表  Ctrl+T+L",self.open_task_list); help_menu.addAction("快捷按键",lambda: ShortcutsDialog(self).exec()); help_menu.addAction("关于软件",lambda: AboutDialog(self).exec())
        self._shortcuts=[]
        for key,handler in (("A",self.previous_image),("D",self.next_image),("Up",self.previous_image),("Down",self.next_image),("Ctrl+S",self.save_current),("Ctrl+Q",self.close)):
            s=QShortcut(QKeySequence(key),self); s.setContext(Qt.ShortcutContext.ApplicationShortcut); s.activated.connect(handler); self._shortcuts.append(s)

    def showEvent(self, event):
        super().showEvent(event)
        if self.main_splitter.sizes()[0] != 360 or self.main_splitter.sizes()[2] != 360:
            self.main_splitter.setSizes([360, max(1, self.main_splitter.width() - 720), 360])

    def _apply_style(self): self.setStyleSheet(idea_stylesheet()+"QListWidget#imageFileList::item:selected, QListWidget#settingsCategories::item:selected { background: #2e436e; } QProgressBar#statusProgress { border-radius: 4px; } QProgressBar#statusProgress::chunk { background: #2e436e; border-radius: 4px; }")
    def _localize_default_group(self):
        if self.settings.label_groups and self.settings.label_groups[0].protected: self.settings.label_groups[0].name="默认标签" if self.settings.language=="zh_CN" else "Default Labels"
    def _apply_language(self): self._localize_default_group(); self.image_panel.set_language(self.settings.language); self.operations_panel.set_language(self.settings.language); self.preset_panel.set_language(self.settings.language); self.refresh_stats()
    def _apply_live_settings(self, settings): self.settings=settings; self.state.settings=settings; self._apply_language(); self._build_language_menu()

    def _build_language_menu(self):
        english = self.settings.language == "en_US"
        labels = ("File", "Edit", "View", "Tools", "Help") if english else ("文件", "编辑", "查看", "工具", "帮助")
        self.menuBar().clear()
        file_menu, edit_menu, view_menu, tools_menu, help_menu = [self.menuBar().addMenu(label) for label in labels]
        self.save_action = file_menu.addAction("Save  Ctrl+S" if english else "保存  Ctrl+S")
        self.save_action.triggered.connect(self.save_current)
        file_menu.addAction("Open  Ctrl+O" if english else "打开  Ctrl+O", self.open_directory)
        file_menu.addAction("Exit  Ctrl+Q" if english else "退出  Ctrl+Q", self.close)
        edit_menu.addAction("Application Settings  Ctrl+A+S" if english else "应用设置  Ctrl+A+S", self.open_settings)
        view_menu.addAction("Task List  Ctrl+T+L" if english else "任务列表  Ctrl+T+L", self.open_task_list)
        tools_menu.addAction("Auto Label  Ctrl+A+L" if english else "自动标注  Ctrl+A+L", self.auto_label_all)
        help_menu.addAction("Shortcuts" if english else "快捷按键", lambda: ShortcutsDialog(self).exec())
        help_menu.addAction("About" if english else "关于软件", lambda: AboutDialog(self).exec())
    def _refresh_toolbar(self): self.save_action.setVisible(True); self.save_action.setEnabled(True)
    def _set_dirty(self,value):
        self.dirty=value; self.status.setText("未保存修改" if value else "")
        if value and self.settings.auto_save and self.project_file: self._auto_save_timer.start(300)
    def refresh_image_list(self): self.image_panel.set_records(self.state.images)
    def refresh_stats(self):
        stats=self.state.statistics(); cur=self.state.current_image; self.status_current_count.setText(f"图标签: {len(cur.annotations) if cur else 0}"); self.status_labeled_count.setText(f"总标注: {stats['labeled_images']}"); self.status_progress_text.setText(f"总进度: {int(stats['percentage'])}%"); self.status_progress.setValue(int(stats['percentage']))
    def select_image(self,row):
        if 0<=row<len(self.state.images): self.state.current_index=row; r=self.state.current_image; self.canvas.load_image(QImage(str(r.path)),r.annotations); self.refresh_stats()
    def previous_image(self): self.select_image(max(0,self.state.current_index-1))
    def next_image(self): self.select_image(min(len(self.state.images)-1,self.state.current_index+1))
    def open_settings(self):
        d=SettingsDialog(self.settings,self)
        if d.exec()==d.DialogCode.Accepted: self._apply_live_settings(d.settings)
    def open_task_list(self): self._show_task_list()
    def _show_task_list(self):
        if not getattr(self,"task_list_dialog",None):
            self.task_list_dialog=TaskListDialog(self.task_manager,self)
            self.task_list_dialog.taskStopped.connect(self._task_stopped)
        self.task_list_dialog.refresh(); self.task_list_dialog.show(); self.task_list_dialog.raise_()
    def _task_stopped(self, name):
        AppDialog.information(name, f"停止{name}成功", self)
    def _close_task_list(self):
        if getattr(self,"task_list_dialog",None): self.task_list_dialog.close()
    def open_directory(self):
        root=QFileDialog.getExistingDirectory(self,"选择数据集目录")
        if not root:return
        path = Path(root)
        try:
            detected, image_dir, annotation_dir = self.project_service.detect_dataset_format(path)
            self.settings.annotation_format = detected
        except ValueError:
            if (path / "classes.txt").exists() and (path / "images").is_dir() and (path / "labels").is_dir():
                self.settings.annotation_format = "yolo"
                image_dir, annotation_dir = path / "images", path / "labels"
            else:
                image_dir,annotation_dir=self.project_service.resolve_dataset_paths(path,self.settings.annotation_format)
        self.settings.image_dir=image_dir; self.settings.annotation_dir=annotation_dir; self._start_dataset_scan(image_dir,annotation_dir)
    def _start_dataset_scan(self,image_dir,annotation_dir):
        self._set_dataset_loading(True); self._dataset_thread=QThread(self); self._dataset_worker=DatasetScanWorker(image_dir,annotation_dir,ProjectSettings.from_dict(self.settings.to_dict())); self.dataset_task_id=self.task_manager.start("打开数据集",self.cancel_dataset_scan,self._count_images(image_dir)); self._show_task_list(); self._dataset_worker.moveToThread(self._dataset_thread); self._dataset_thread.started.connect(self._dataset_worker.run); self._dataset_worker.progress.connect(self._dataset_scan_progress); self._dataset_worker.finished.connect(self._dataset_scan_finished); self._dataset_worker.failed.connect(self._dataset_scan_failed); self._dataset_worker.finished.connect(self._dataset_thread.quit); self._dataset_worker.failed.connect(self._dataset_thread.quit); self._dataset_thread.finished.connect(self._dataset_thread_finished); self._dataset_thread.start()
    def _dataset_scan_progress(self,current,total):
        p=int(current/total*100) if total else 0; self.status.setText(f"正在加载数据集: {current}/{total}"); self.status_progress_text.setText(f"加载: {p}%"); self.status_progress.setValue(p); self.task_manager.update(self.dataset_task_id,p,current,total)
    def _dataset_scan_finished(self,records):
        if self._dataset_worker and self._dataset_worker.cancelled:return
        self.state.images=records; self.state.current_index=0 if records else -1; self.refresh_image_list(); self._set_dataset_loading(False); self._close_task_list()
        if self.isVisible(): AppDialog.information("打开数据集","数据集加载成功",self)
    def _dataset_scan_failed(self,message):
        self._set_dataset_loading(False); self._close_task_list()
        if self.isVisible(): AppDialog.information("打开数据集失败",message,self)
    def _dataset_thread_finished(self): self.task_manager.finish(self.dataset_task_id); self.dataset_task_id=None; self._dataset_thread=None; self._dataset_worker=None
    def cancel_dataset_scan(self):
        if self._dataset_worker:
            self._dataset_worker.cancelled = True
            self._set_dataset_loading(False)
            self._close_task_list()
    def _set_dataset_loading(self,active): self.canvas.setEnabled(not active); self.image_panel.setEnabled(not active); self.preset_panel.setEnabled(not active)
    @staticmethod
    def _count_images(directory): return sum(1 for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"})
    def save_current(self):
        if not self.project_file or not self.state.current_image:
            self.dirty=False; return
        self._save_thread=QThread(self); self._save_worker=SaveWorker(self.project_file,self.state.current_image.path,list(self.canvas.annotations),ProjectSettings.from_dict(self.settings.to_dict())); self._save_worker.moveToThread(self._save_thread); self._save_thread.started.connect(self._save_worker.run); self._save_worker.finished.connect(self._save_thread.quit); self._save_worker.finished.connect(lambda: setattr(self,'_save_thread',None)); self._save_thread.finished.connect(self._save_thread.deleteLater); self._save_thread.start(); self.dirty=False
    def auto_label_all(self): AppDialog.information("自动标注","自动标注任务已加入任务列表",self)
    def _auto_label_failed(self, message): self.status.setText(""); AppDialog.information("自动标注失败", message, self)
    def open_statistics(self):
        if not self.state.images:
            AppDialog.information("数据统计", "请先打开数据集", self)
            return
        StatsDialog(self.state.statistics(), self.settings.language, self).exec()

    def open_conversion(self):
        dialog = ConversionDialog(self.settings.label_presets, self)
        if dialog.exec() == dialog.DialogCode.Accepted and dialog.options is not None:
            self._start_conversion(dialog.options)

    def _start_conversion(self, options):
        self._conversion_thread = QThread(self)
        self._conversion_worker = ConversionWorker(options)
        total = sum(1 for p in options.source_path.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        self.conversion_task_id = self.task_manager.start("数据集转换", self.cancel_conversion, total)
        self._conversion_worker.moveToThread(self._conversion_thread)
        self._conversion_thread.started.connect(self._conversion_worker.run)
        self._conversion_worker.progress.connect(self._conversion_progress)
        self._conversion_worker.completed.connect(self._conversion_finished)
        self._conversion_worker.failed.connect(self._conversion_failed)
        self._conversion_worker.completed.connect(self._conversion_thread.quit)
        self._conversion_worker.failed.connect(self._conversion_thread.quit)
        self._conversion_thread.finished.connect(self._conversion_thread_finished)
        self._show_task_list(); self._conversion_thread.start()

    def _conversion_progress(self, current, total):
        percent = int(current / total * 100) if total else 0
        self.task_manager.update(self.conversion_task_id, percent, current, total)

    def _conversion_finished(self, report):
        if report.failed:
            self._conversion_failed("; ".join(report.errors[:3]) or "数据集转换失败")
        else:
            AppDialog.information("数据集转换", f"转换完成，共处理 {report.succeeded} 个文件", self)

    def _conversion_failed(self, message):
        AppDialog.information("数据集转换失败", message, self)

    def _conversion_thread_finished(self):
        self.task_manager.finish(self.conversion_task_id)
        self.conversion_task_id = None
        self._conversion_thread = None
        self._conversion_worker = None

    def cancel_conversion(self):
        if self._conversion_worker: self._conversion_worker.cancelled = True
    def auto_label_all(self):
        if not self.state.images:
            AppDialog.information("自动标注", "请先打开数据集", self)
            return
        if not self.settings.onnx_model_path:
            AppDialog.information("自动标注", "请先在应用设置中选择 ONNX 模型", self)
            return
        self._auto_thread = QThread(self)
        self._auto_worker = AutoLabelWorker(self.state.images, ProjectSettings.from_dict(self.settings.to_dict()))
        self.auto_task_id = self.task_manager.start("自动标注", self.cancel_auto_label, len(self.state.images))
        self._auto_worker.moveToThread(self._auto_thread)
        self._auto_thread.started.connect(self._auto_worker.run)
        self._auto_worker.progress.connect(self._auto_progress)
        self._auto_worker.finished.connect(self._auto_finished)
        self._auto_worker.failed.connect(self._auto_failed)
        self._auto_worker.finished.connect(self._auto_thread.quit)
        self._auto_worker.failed.connect(self._auto_thread.quit)
        self._auto_thread.finished.connect(self._auto_thread_finished)
        self._set_dataset_loading(True)
        self._show_task_list()
        self._auto_thread.start()

    def _auto_progress(self, current, total):
        percent = int(current / total * 100) if total else 0
        self.task_manager.update(self.auto_task_id, percent, current, total)

    def _auto_finished(self):
        self.refresh_image_list()
        self.refresh_stats()
        self._set_dirty(True)
        AppDialog.information("自动标注", "自动标注完成", self)

    def _auto_failed(self, message):
        AppDialog.information("自动标注失败", message, self)

    def cancel_auto_label(self):
        if self._auto_worker:
            self._auto_worker.cancelled = True

    def _auto_thread_finished(self):
        self._set_dataset_loading(False)
        self.task_manager.finish(self.auto_task_id)
        self.auto_task_id = None
        self._auto_thread = None
        self._auto_worker = None
    def open_label_groups(self):
        dialog = LabelGroupsDialog(self.settings.label_groups, self, self.settings.language)
        if dialog.exec() == dialog.DialogCode.Accepted:
            self.settings.label_groups = dialog.groups
            selected = self.preset_panel.group_combo.currentText() if hasattr(self.preset_panel, "group_combo") else self.settings.label_groups[0].name
            self.preset_panel.set_groups(self.settings.label_groups, selected)
            self.preset_panel.set_presets(self.preset_panel.selected_group().presets if self.preset_panel.selected_group() else self.settings.label_presets)

    def open_image_filter(self):
        dialog = ImageFilterDialog(self.image_panel.search.text(), self.image_panel.selected_status(), self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            query, status = dialog.values()
            self.image_panel.search.setText(query)
            self.image_panel.set_status_filter(status)
            self.refresh_image_list()

    def open_crosshair(self):
        dialog = CrosshairDialog(self.settings.crosshair_line_width, self.settings.crosshair_color, self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            width, color = dialog.changed if False else (dialog.line_width.value(), dialog._color.name())
            self.settings.crosshair_line_width = width
            self.settings.crosshair_color = color
            self.canvas.set_crosshair_settings(width, color)
