from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication, QDialog, QMenu, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QToolButton, QVBoxLayout, QWidget
from .common_dialogs import AppDialog
from .form_layout import BUTTON_TOP_SPACING, configure_buttons, set_confirm_button, set_content_margins, size_buttons


class HistoryDialog(QDialog):
    historyChanged = Signal(list)

    def __init__(self, paths: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("管理历史")
        # Rows show folder names; width fits the longest name within 85% of
        # the main window so names stay fully readable.
        self.setFixedHeight(380)
        metrics = self.fontMetrics()
        longest = max((metrics.horizontalAdvance(Path(p).name or p) for p in paths), default=0)
        wanted = longest + 120  # dialog margins + row padding + menu button
        max_width = 640
        window = parent.window() if parent is not None else None
        if window is not None:
            max_width = max(420, int(window.width() * 0.85))
        self.setFixedWidth(max(420, min(wanted, max_width)))
        self.paths = list(paths)
        self.pending_deleted: set[str] = set()
        self._open_requested: str | None = None
        self.filtered_paths: list[str] = []

        self.search = QLineEdit()
        self.search.setFixedHeight(30)
        self.search.setPlaceholderText("搜索名称或路径")
        self.search.textChanged.connect(self._refresh_list)
        self.list = QListWidget()
        self.list.setObjectName("historyList")
        self.list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list.setSpacing(0)
        self.list.currentItemChanged.connect(self._refresh_item_styles)
        self.list.itemDoubleClicked.connect(self._open_item)

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
        layout.addSpacing(15)
        layout.addWidget(self.list, 1)
        layout.addSpacing(BUTTON_TOP_SPACING)
        layout.addLayout(buttons)
        self.list.setStyleSheet(
            "QListWidget { background: #282A2F; border: 1px solid #3C4148; border-radius: 6px; padding: 6px; outline: 0; }"
            "QListWidget::item { border: none; background: transparent; border-radius: 5px; margin: 0 0 6px 0; }"
        )

        self._refresh_list()


    def _open_item(self, item) -> None:
        row = self.list.row(item)
        if 0 <= row < len(self.filtered_paths):
            self.historyChanged.emit([p for p in self.paths if p not in self.pending_deleted])
            self._open_requested = self.filtered_paths[row]
            self.accept()

    def _refresh_item_styles(self, current=None, previous=None) -> None:
        for index in range(self.list.count()):
            item = self.list.item(index)
            row = self.list.itemWidget(item)
            if row is None:
                continue
            row.setProperty("selected", item is current)
            row.style().unpolish(row); row.style().polish(row)

    def _refresh_list(self) -> None:
        query = self.search.text().casefold().strip()
        self.filtered_paths = [
            path for path in self.paths
            if path not in self.pending_deleted
            and (not query or self._fuzzy_match(query, path))
        ]
        self.list.clear()
        from src.app_paths import resource_path
        icon_path = resource_path("icons/ic_more.png")
        for path in self.filtered_paths:
            item = QListWidgetItem(self.list)
            item.setSizeHint(QSize(0, 36))  # 30px card + 6px margin below
            row = QWidget()
            row.setObjectName("historyItem")
            row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            row.setStyleSheet(
                "QWidget#historyItem { background: #2F3237; border: 1px solid #3E424A; border-left: 3px solid #3E424A; border-radius: 5px; }"
                "QWidget#historyItem:hover { background: #383C42; border-left: 3px solid #6A84B8; }"
                "QWidget#historyItem[selected=\"true\"] { background: #31436B; border: 1px solid #6A84B8; border-left: 3px solid #7FA3E0; }"
            )
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(11, 0, 7, 0)
            row_layout.setSpacing(8)
            full = QLabel(Path(path).name or path)
            full.setToolTip(path)
            full.setStyleSheet("QLabel { color: #C6CBD3; background: transparent; font-size: 12px; }")
            menu_button = QToolButton()
            menu_button.setObjectName("historyMenuButton")
            menu_button.setIcon(QIcon(str(icon_path)))
            menu_button.setIconSize(QSize(12, 12))
            menu_button.setToolTip("更多操作")
            menu_button.setFixedSize(20, 20)
            menu_button.setCursor(Qt.CursorShape.PointingHandCursor)
            menu_button.setStyleSheet("QToolButton#historyMenuButton { background: transparent; border: none; padding: 0; border-radius: 4px; } QToolButton#historyMenuButton:hover { background: #3C4046; }")
            menu_button.clicked.connect(lambda checked=False, value=path, btn=menu_button: self._show_item_menu(btn, value))
            row_layout.addWidget(full, 1)
            row_layout.addWidget(menu_button)
            self.list.setItemWidget(item, row)
        self._refresh_item_styles(self.list.currentItem())

    @staticmethod
    def _fuzzy_match(query: str, value: str) -> bool:
        """Match a query as a case-insensitive substring of the full path."""
        query = query.casefold().strip()
        value = value.casefold()
        return not query or query in value

    def _build_item_menu(self, path: str) -> QMenu:
        menu = QMenu(self)
        menu.addAction("加载目录", lambda: self._load_dataset(path, check_exists=True))
        menu.addAction("浏览目录", lambda: self._browse_directory(path))
        menu.addAction("复制路径", lambda: QApplication.clipboard().setText(path))
        menu.addSeparator()
        menu.addAction("删除历史", lambda: self._remove_path(path))
        return menu

    def _browse_directory(self, path: str) -> None:
        if not self._warn_if_missing(path):
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _warn_if_missing(self, path: str) -> bool:
        """Prompt when the directory no longer exists; True means proceed."""
        if Path(path).is_dir():
            return True
        AppDialog.information("提示", f"目录不存在：{path}", self)
        return False

    def _load_dataset(self, path: str, check_exists: bool = False) -> None:
        if check_exists and not self._warn_if_missing(path):
            return
        self.historyChanged.emit([p for p in self.paths if p not in self.pending_deleted])
        self._open_requested = path
        self.accept()

    def _show_item_menu(self, button: QToolButton, path: str) -> None:
        self._build_item_menu(path).exec(button.mapToGlobal(QPoint(0, button.height())))

    def _remove_path(self, path: str) -> None:
        if path not in self.paths:
            return
        self.pending_deleted.add(path)
        self._refresh_list()

    def open_requested(self) -> str | None:
        return self._open_requested

    def _confirm(self) -> None:
        remaining = [path for path in self.paths if path not in self.pending_deleted]
        self.historyChanged.emit(remaining)
        self.accept()
