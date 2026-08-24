from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


class ModelStatusBar(QStatusBar):
    def set_content_layout(self, layout):
        self._content_layout = layout
        self._content_host = QWidget(self)
        self._content_host.setLayout(layout)
        self.addPermanentWidget(self._content_host, 1)

    def layout(self):
        return getattr(self, "_content_layout", super().layout())


class FileTextDialog(QDialog):
    def __init__(self, title: str, content: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.english = getattr(getattr(parent, "settings", None), "language", "zh_CN") == "en_US"
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(content)
        editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        editor.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        editor.customContextMenuRequested.connect(lambda position: self._show_text_menu(editor, position))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.addWidget(editor)
        self._resize_to_content(editor, content)

    def _resize_to_content(self, editor: QPlainTextEdit, content: str) -> None:
        metrics = QFontMetrics(editor.font())
        lines = content.splitlines() or [""]
        longest_line = max((metrics.horizontalAdvance(line) for line in lines), default=0)
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        max_width = min(900, available.width() - 80) if available is not None else 900
        max_height = int(available.height() * 0.8) if available is not None else 700
        self.resize(
            max(320, min(max_width, longest_line + 52)),
            max(120, min(max_height, len(lines) * metrics.lineSpacing() + 52)),
        )

    def _show_text_menu(self, field: QPlainTextEdit, position) -> None:
        menu = QMenu(self)
        copy_action = menu.addAction("Copy" if self.english else "复制")
        select_all_action = menu.addAction("Select All" if self.english else "全选")
        chosen = menu.exec(field.mapToGlobal(position))
        if chosen is copy_action:
            field.copy()
        elif chosen is select_all_action:
            field.selectAll()


class FileInfoDialog(QDialog):
    def __init__(self, title: str, rows: list[tuple[str, str]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(520)
        self.english = getattr(getattr(parent, "settings", None), "language", "zh_CN") == "en_US"
        grid = QGridLayout(self)
        grid.setContentsMargins(18, 18, 18, 18)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(10)
        for row, (label, value) in enumerate(rows):
            key = QLabel(label)
            key.setStyleSheet("font-weight: 600; color: #AEB4C0;")
            val = QLineEdit(value)
            val.setReadOnly(True)
            val.setFrame(False)
            val.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            val.customContextMenuRequested.connect(lambda position, field=val: self._show_value_menu(field, position))
            val.setStyleSheet("QLineEdit { background: transparent; border: none; color: #D7DAE0; padding: 0; }")
            grid.addWidget(key, row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(val, row, 1)
        grid.setColumnStretch(1, 1)

    def _show_value_menu(self, field: QLineEdit, position) -> None:
        menu = QMenu(self)
        copy_action = menu.addAction("Copy" if self.english else "复制")
        select_all_action = menu.addAction("Select All" if self.english else "全选")
        chosen = menu.exec(field.mapToGlobal(position))
        if chosen is copy_action:
            field.copy()
        elif chosen is select_all_action:
            field.selectAll()
