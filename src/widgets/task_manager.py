from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QObject, Signal


@dataclass
class ActiveTask:
    task_id: int
    name: str
    progress: int = 0
    current: int = 0
    total: int = 0
    cancel: Callable[[], None] | None = None


class TaskManager(QObject):
    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._next_id = 1
        self._tasks: dict[int, ActiveTask] = {}

    def start(self, name: str, cancel: Callable[[], None] | None = None, total: int = 0) -> int:
        task_id = self._next_id
        self._next_id += 1
        self._tasks[task_id] = ActiveTask(task_id, name, total=max(0, int(total)), cancel=cancel)
        self.changed.emit()
        return task_id

    def update(self, task_id: int | None, progress: int, current: int | None = None, total: int | None = None) -> None:
        task = self._tasks.get(task_id)
        if task is not None:
            task.progress = max(0, min(100, int(progress)))
            if current is not None:
                task.current = max(0, int(current))
            if total is not None:
                task.total = max(0, int(total))
            self.changed.emit()

    def finish(self, task_id: int | None) -> None:
        if task_id is not None and self._tasks.pop(task_id, None) is not None:
            self.changed.emit()

    def tasks(self) -> list[ActiveTask]:
        return list(self._tasks.values())

    def cancel(self, task_id: int) -> None:
        task = self._tasks.get(task_id)
        if task is not None and task.cancel is not None:
            task.cancel()
