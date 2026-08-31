from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QAbstractItemView, QComboBox, QLabel, QLineEdit, QListView, QVBoxLayout, QWidget

from src.models.project import ImageRecord


class ImageFileModel(QAbstractListModel):
    recordsFetched = Signal(object)
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.records: list[ImageRecord] = []
        self._page_loader: Callable[[int, int], list[ImageRecord]] | None = None
        self._total_count = 0
        self._page_size = 500

    @staticmethod
    def _text(record: ImageRecord) -> str:
        return record.path.name

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.records)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.records):
            return None
        record = self.records[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return self._text(record)
        if role == Qt.ItemDataRole.UserRole:
            return record.path
        return None

    def set_records(self, records: list[ImageRecord]) -> None:
        self._page_loader = None
        self._total_count = len(records)
        self.beginResetModel()
        self.records = list(records)
        self.endResetModel()

    def set_paged_records(self, records: list[ImageRecord], total_count: int, loader: Callable[[int, int], list[ImageRecord]]) -> None:
        self._page_loader = loader
        self._total_count = max(len(records), int(total_count))
        self.beginResetModel()
        self.records = list(records)
        self.endResetModel()

    def set_total_count(self, total_count: int) -> None:
        self._total_count = max(len(self.records), int(total_count))

    def canFetchMore(self, parent=QModelIndex()) -> bool:
        return not parent.isValid() and self._page_loader is not None and len(self.records) < self._total_count

    def fetchMore(self, parent=QModelIndex()) -> None:
        if not self.canFetchMore(parent):
            return
        page = self._page_loader(len(self.records), self._page_size)
        self.append_records(page)

    def append_records(self, records: list[ImageRecord]) -> None:
        if not records:
            return
        start = len(self.records)
        end = start + len(records) - 1
        self.beginInsertRows(QModelIndex(), start, end)
        self.records.extend(records)
        self.endInsertRows()
        self.recordsFetched.emit(records)

    def update_record(self, record: ImageRecord) -> None:
        row = next((i for i, item in enumerate(self.records) if item.path == record.path), -1)
        if row >= 0:
            self.records[row] = record
            index = self.index(row, 0)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.UserRole])


class ImageFileList(QListView):
    previousRequested = Signal()
    nextRequested = Signal()
    bottom_spacing = 40

    def updateGeometries(self) -> None:
        super().updateGeometries()
        model = self.model()
        if model is None or model.rowCount() <= 0:
            return
        row_height = self.sizeHintForRow(0)
        if row_height <= 0:
            return
        row_extent = row_height + self.spacing() * 2
        content_height = model.rowCount() * row_extent
        maximum = max(0, content_height - self.viewport().height() + self.bottom_spacing)
        self.verticalScrollBar().setMaximum(maximum)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_A):
            self.previousRequested.emit()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_D):
            self.nextRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def setCurrentRow(self, row: int) -> None:
        self.setCurrentIndex(self.model().index(row, 0) if self.model() and row >= 0 else QModelIndex())

    def currentRow(self) -> int:
        return self.currentIndex().row()


class ImageListPanel(QWidget):
    imageSelected = Signal(int)
    recordsFetched = Signal(object)
    contextMenuRequested = Signal(object)
    filtersChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("projectPanel")
        self.title_label = QLabel("")
        self.hint_label = QLabel("")
        self.search = QLineEdit()
        self.search.setPlaceholderText("\u641c\u7d22\u56fe\u7247...")
        self.label_filter = ""
        self.status = QComboBox()
        self.status.addItem("All", "all")
        self.status.addItem("Labeled", "labeled")
        self.status.addItem("Unlabeled", "unlabeled")
        self.list = ImageFileList()
        self.list_model = ImageFileModel(self.list)
        self.list.setModel(self.list_model)
        self.list.setObjectName("imageFileList")
        self.list.setFrameShape(QListView.Shape.NoFrame)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.setLineWidth(0)
        self.list.setSpacing(2)
        self.list.setUniformItemSizes(True)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.setStyleSheet(
            "QListView#imageFileList, QListView#imageFileList:focus { background: #282A2F; border: 1px solid #3C4148; border-radius: 6px; padding: 6px; outline: 0; }"
            "QListView#imageFileList::item { height: 30px; padding: 0 5px; margin: 0 0 4px 0; background: #2F3237; color: #C6CBD3; border: 1px solid #3E424A; border-left: 3px solid #3E424A; border-radius: 5px; outline: 0; }"
            "QListView#imageFileList::item:hover { background: #383C42; color: #FFFFFF; border-left: 3px solid #6A84B8; }"
            "QListView#imageFileList::item:selected, QListView#imageFileList::item:selected:focus { background: #31436B; color: #FFFFFF; font-weight: 600; border: 1px solid #6A84B8; border-left: 3px solid #7FA3E0; outline: 0; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(self.search)
        self.status.hide()
        layout.addWidget(self.list)
        self.search.textChanged.connect(self.filtersChanged)
        self.status.currentIndexChanged.connect(self.filtersChanged)
        self.list.selectionModel().currentChanged.connect(lambda current, previous: self.imageSelected.emit(current.row()))
        self.list.customContextMenuRequested.connect(self._context_menu_requested)
        self.list_model.recordsFetched.connect(self._records_fetched)
        self.records: list[ImageRecord] = []

    def _records_fetched(self, records: list[ImageRecord]) -> None:
        self.records.extend(records)
        self.recordsFetched.emit(records)

    def _context_menu_requested(self, position) -> None:
        index = self.list.indexAt(position)
        if not index.isValid() or not 0 <= index.row() < len(self.list_model.records):
            return
        self.list.setCurrentIndex(index)
        self.contextMenuRequested.emit(self.list_model.records[index.row()])

    def set_language(self, language: str) -> None:
        self.search.setPlaceholderText("Search images..." if language == "en_US" else "\u641c\u7d22\u56fe\u7247...")

    def set_records(self, records: list[ImageRecord]) -> None:
        self.records = records
        self.list_model.set_records(records)
        self.list.viewport().update()

    def append_records(self, records: list[ImageRecord]) -> None:
        if not records:
            return
        self.list_model.append_records(records)
        self.list.viewport().update()

    def set_paged_records(self, records: list[ImageRecord], total_count: int, loader: Callable[[int, int], list[ImageRecord]]) -> None:
        self.records = list(records)
        self.list_model.set_paged_records(records, total_count, loader)
        self.list.viewport().update()

    def set_total_count(self, total_count: int) -> None:
        self.list_model.set_total_count(total_count)

    def select_record(self, record: ImageRecord | None) -> None:
        """Select by path without emitting a transient invalid row."""
        row = next((index for index, item in enumerate(self.records) if item.path == getattr(record, "path", None)), -1)
        blocker = self.list.blockSignals(True)
        self.list.setCurrentRow(row)
        self.list.blockSignals(blocker)

    def update_record(self, record: ImageRecord) -> None:
        row = next(
            (index for index, item in enumerate(self.records) if item.path == record.path),
            -1,
        )
        if row < 0:
            return
        self.list_model.update_record(record)

    def selected_status(self) -> str:
        return self.status.currentData()

    def set_label_filter(self, value: str) -> None:
        self.label_filter = value.strip()

    def selected_label(self) -> str:
        return self.label_filter
