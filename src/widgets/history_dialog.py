from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QToolButton, QVBoxLayout, QWidget
from .form_layout import BUTTON_TOP_SPACING, configure_buttons, set_confirm_button, set_content_margins, size_buttons


class HistoryDialog(QDialog):
    historyChanged = Signal(list)

    def __init__(self, paths: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("管理历史")
        self.setFixedSize(400, 320)
        self.paths = list(paths)
        self.pending_deleted: set[str] = set()
        self.filtered_paths: list[str] = []

        self.search = QLineEdit()
        self.search.setPlaceholderText("\u641c\u7d22\u540d\u79f0")
        self.search.setPlaceholderText("模糊搜索历史路径")
        self.search.setPlaceholderText("\u641c\u7d22\u540d\u79f0")
        self.search.textChanged.connect(self._refresh_list)
        self.list = QListWidget()
        self.list.setObjectName("historyList")
        self.list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list.setSpacing(6)
        self.list.currentItemChanged.connect(self._refresh_item_styles)

        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        confirm = QPushButton("确认")
        confirm.clicked.connect(self._confirm)
        buttons = configure_buttons(QHBoxLayout())
        buttons.addStretch()
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)
        size_buttons(cancel, confirm)
        set_confirm_button(confirm)

        layout = QVBoxLayout(self)
        set_content_margins(layout)
        layout.setSpacing(0)
        layout.addWidget(self.search)
        layout.addSpacing(20)
        layout.addWidget(self.list, 1)
        layout.addSpacing(BUTTON_TOP_SPACING)
        layout.addLayout(buttons)
        self.list.setStyleSheet(
            "QListWidget { background: #25272A; border: 1px solid #464A50; border-radius: 5px; padding: 5px; }"
            "QListWidget::item { height: 30px; padding: 0; margin: 0 0 2px 0; background: #35383D; color: #FFFFFF; border: 2px solid transparent; border-radius: 5px; }"
            "QListWidget::item:hover { background: #41454C; color: #FFFFFF; border: 2px solid #FFFFFF; }"
            "QListWidget::item:selected, QListWidget::item:selected:active { background: #2e436e; color: #FFFFFF; font-weight: 600; border: 2px solid #FFFFFF; }"
        )

        self._refresh_list()

    def _refresh_item_styles(self, current=None, previous=None) -> None:
        for index in range(self.list.count()):
            item = self.list.item(index)
            row = self.list.itemWidget(item)
            if row is None:
                continue
            selected = item is current
            row.setStyleSheet(
                "QWidget#historyItem { background: %s; border: 2px solid %s; border-radius: 5px; }"
                % ("#2e436e" if selected else "#35383D", "#FFFFFF" if selected else "transparent")
            )

    def _refresh_list(self) -> None:
        query = self.search.text().casefold().strip()
        self.filtered_paths = [
            path for path in self.paths
            if path not in self.pending_deleted
            and (not query or self._fuzzy_match(query, Path(path).name))
        ]
        self.list.clear()
        icon_path = Path(__file__).resolve().parents[2] / "icons" / "ic_delete.png"
        for path in self.filtered_paths:
            item = QListWidgetItem(self.list)
            item.setSizeHint(QSize(0, 30))
            row = QWidget()
            row.setObjectName("historyItem")
            row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            row.setStyleSheet("QWidget#historyItem { background: transparent; border: none; }")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 0, 4, 0)
            row_layout.setSpacing(8)
            label = QLabel(Path(path).name or path)
            label.setToolTip(path)
            label.setStyleSheet("QLabel { color: #D7DAE0; background: transparent; }")
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            delete = QToolButton()
            delete.setObjectName("historyDeleteButton")
            delete.setIcon(QIcon(str(icon_path)))
            delete.setIconSize(QSize(10, 10))
            delete.setToolTip("删除")
            delete.setFixedSize(20, 20)
            delete.setStyleSheet("QToolButton#historyDeleteButton { background: transparent; border: none; padding: 0; } QToolButton#historyDeleteButton:hover, QToolButton#historyDeleteButton:pressed { background: transparent; border: none; }")
            delete.clicked.connect(lambda checked=False, value=path: self._remove_path(value))
            row_layout.addWidget(label, 1)
            row_layout.addWidget(delete)
            self.list.setItemWidget(item, row)
        self._refresh_item_styles(self.list.currentItem())

    @staticmethod
    def _fuzzy_match(query: str, value: str) -> bool:
        """Match a query as a case-insensitive continuous substring."""
        query = query.casefold().strip()
        value = value.casefold()
        return not query or query in value

    def _remove_path(self, path: str) -> None:
        if path not in self.paths:
            return
        self.pending_deleted.add(path)
        self._refresh_list()

    def _confirm(self) -> None:
        remaining = [path for path in self.paths if path not in self.pending_deleted]
        self.historyChanged.emit(remaining)
        self.accept()
