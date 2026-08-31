from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .task_manager import ActiveTask, TaskManager


class TaskListDialog(QDialog):
    taskStopped = Signal(str)
    # Dataset scanning remains managed by TaskManager, but it is intentionally
    # omitted from the user-facing task list. This view is for long-running
    # labeling and conversion operations only.
    VISIBLE_TASKS = frozenset({"自动标注", "数据集转换"})

    def __init__(self, manager: TaskManager, parent=None, language: str = "zh_CN") -> None:
        super().__init__(parent)
        # MainWindow's stylesheet is not reliably propagated to a separately
        # created top-level dialog on Windows. Style only this dialog surface
        # so its first frame cannot use the system white background.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(palette.ColorRole.Window, QColor("#2B2D30"))
        self.setPalette(palette)
        self.setStyleSheet(
            "QDialog { background-color: #2B2D30; color: #D7DAE0; }"
            "QLabel { color: #D7DAE0; background: transparent; }"
        )
        self.manager = manager
        self.language = language
        self.setMinimumWidth(360)
        self.rows = QVBoxLayout(self)
        self.rows.setContentsMargins(18, 18, 18, 18)
        self.rows.setSpacing(8)
        self._task_rows: dict[int, QWidget] = {}
        self._empty_state: QLabel | None = None
        self.manager.changed.connect(self.refresh)
        self.set_language(language)
        self.refresh()

    def set_language(self, language: str) -> None:
        self.language = language
        self.setWindowTitle("Task Management" if language == "en_US" else "管理任务")
        if self._empty_state is not None:
            self._empty_state.setText("No tasks" if language == "en_US" else "空任务")
        for row in self._task_rows.values():
            button = getattr(row, "_task_stop", None)
            if button is not None:
                button.setToolTip("Stop" if language == "en_US" else "停止")
                button.setAccessibleName("Stop" if language == "en_US" else "停止")
        self.refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        parent = self.parentWidget()
        if parent is not None:
            # Task monitoring must not block the workbench. Dataset loading
            # runs in a worker thread, so the user can keep using menus and
            # inspect already indexed images while this panel is visible.
            self.setWindowModality(Qt.WindowModality.NonModal)
            frame = parent.window().frameGeometry()
            target = frame.center() - self.rect().center()
            self.move(target)

    def _task_name(self, name: str) -> str:
        if self.language != "en_US":
            return name
        return {"打开数据集": "Open Dataset", "数据集转换": "Dataset Conversion", "自动标注": "Auto Label"}.get(name, name)

    def refresh(self) -> None:
        tasks = [task for task in self.manager.tasks() if task.name in self.VISIBLE_TASKS]
        if not tasks:
            self._remove_stale_rows(set())
            if self._empty_state is None:
                self._empty_state = QLabel("No tasks" if self.language == "en_US" else "空任务")
                self._empty_state.setObjectName("statusMuted")
                self.rows.addWidget(self._empty_state)
            return

        if self._empty_state is not None:
            self.rows.removeWidget(self._empty_state)
            self._empty_state.deleteLater()
            self._empty_state = None

        task_ids = {task.task_id for task in tasks}
        self._remove_stale_rows(task_ids)
        for task in tasks:
            row = self._task_rows.get(task.task_id)
            if row is None:
                row = self._create_task_row(task)
                self._task_rows[task.task_id] = row
                self.rows.addWidget(row)
            else:
                self._update_task_row(row, task)

    def _remove_stale_rows(self, active_ids: set[int]) -> None:
        for task_id in list(self._task_rows):
            if task_id in active_ids:
                continue
            row = self._task_rows.pop(task_id)
            self.rows.removeWidget(row)
            row.hide()
            # Keep the task row parented until deferred deletion. Detaching a
            # widget here can briefly promote it to a top-level window at the
            # screen origin.
            row.deleteLater()

    def _create_task_row(self, task: ActiveTask) -> QWidget:
        row = QWidget()
        row.setObjectName("taskProgressItem")
        row.setFixedHeight(40)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(8, 0, 6, 0)
        label = QLabel()
        label.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(label, 1)

        stop = QPushButton()
        stop.setObjectName("taskStopButton")
        from src.app_paths import resource_path
        stop.setIcon(QIcon(str(resource_path("icons/ic_stop.png"))))
        stop.setIconSize(QSize(15, 15))
        stop.setFixedSize(20, 20)
        stop.setToolTip("\u505c\u6b62")
        stop.setAccessibleName("\u505c\u6b62")
        stop.setStyleSheet(
            "QPushButton#taskStopButton { background: transparent; border: none; padding: 0; } "
            "QPushButton#taskStopButton:hover, QPushButton#taskStopButton:pressed { background: transparent; border: none; }"
        )
        stop.clicked.connect(lambda _=False, task_id=task.task_id, name=task.name: self._stop_task(task_id, name))
        layout.addWidget(stop)

        row._task_label = label
        row._task_stop = stop
        self._update_task_row(row, task)
        return row

    def _update_task_row(self, row: QWidget, task: ActiveTask) -> None:
        count = f"{task.current}/{task.total}"
        task_name = self._task_name(task.name)
        text = f"{task_name}  {count}  {task.progress}%" if task.total or task.current else f"{task_name}  {task.progress}%"
        row._task_label.setText(text)

        progress = max(0, min(100, int(task.progress)))
        ratio = progress / 100
        row.setStyleSheet(
            "QWidget#taskProgressItem { "
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            f"stop:0 #2e436e, stop:{ratio:.4f} #2e436e, "
            f"stop:{ratio:.4f} #25262A, stop:1 #25262A); "
            "border: 1px solid #3c4d68; border-radius: 5px; }"
        )

    def _stop_task(self, task_id: int, name: str) -> None:
        self.manager.cancel(task_id)
        self.taskStopped.emit(name)
        # Keep the dialog open so other simultaneous tasks remain visible.
        self.refresh()
