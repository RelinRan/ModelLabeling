from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QStyle, QVBoxLayout, QWidget

from .i18n import text


class StatItem(QFrame):
    def __init__(self, label: str, value: str, summary: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("statItem")
        self.setProperty("summary", summary)
        self.setFixedWidth(150)
        self.setFixedHeight(30)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        label_view = QLabel(label)
        label_view.setObjectName("statItemLabel")
        label_view.setFixedWidth(75)
        label_view.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        value_view = QLabel(value)
        value_view.setObjectName("statItemValue")
        value_view.setFixedWidth(75)
        value_view.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label_view)
        layout.addWidget(value_view)


class StatsDialog(QDialog):
    def __init__(self, stats: dict, language: str = "zh_CN", parent=None) -> None:
        super().__init__(parent)
        self.language = language
        self.setObjectName("statisticsDialog")
        self.setWindowTitle(text("statistics", language))

        panel = QFrame()
        panel.setObjectName("statsCard")
        grid_host = QWidget()
        grid_host.setObjectName("statsGridHost")
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        grid.setAlignment(Qt.AlignmentFlag.AlignTop)

        names = ("图片总数", "已标注", "完成度", "标签总数") if language == "zh_CN" else ("Total images", "Labeled", "Progress", "Total labels")
        values = (stats["total_images"], stats["labeled_images"], f"{stats['percentage']:.1f}%", stats["total_labels"])
        items = list(zip(names, (str(value) for value in values)))
        counts = list(stats["label_counts"].items())
        items.extend((str(name), str(count)) for name, count in counts)
        if not counts:
            items.append((text("none", language), "--"))

        item_count = len(items)
        columns = 1 if item_count <= 2 else 2 if item_count <= 6 else 3 if item_count <= 12 else 4
        for index, (label, value) in enumerate(items):
            row, column = divmod(index, columns)
            grid.addWidget(StatItem(label, value, summary=index < 4), row, column)

        scroll = QScrollArea()
        self.scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(grid_host)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.addWidget(scroll)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(panel)

        rows = (item_count + columns - 1) // columns
        content_width = columns * 150 + max(0, columns - 1) * grid.horizontalSpacing()
        content_height = rows * 30 + max(0, rows - 1) * grid.verticalSpacing()
        grid_host.setMinimumSize(content_width, content_height)

        screen = self.screen() or QApplication.primaryScreen()
        available_height = screen.availableGeometry().height() if screen else 900
        maximum_height = max(200, int(available_height * 0.8))
        desired_height = content_height + 64
        needs_scroll = desired_height > maximum_height
        scrollbar_width = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent) if needs_scroll else 0
        desired_width = content_width + 64 + scrollbar_width
        self.resize(desired_width, min(desired_height, maximum_height))

        self.setStyleSheet("""
            QFrame#statsCard, QWidget#statsGridHost, QScrollArea { background: transparent; border: none; }
            QFrame#statItem { background: #252a31; border: 1px solid #3c4652; border-radius: 5px; }
            QLabel#statItemLabel { background: #303640; color: #b9c4cf; padding: 0 6px; border-top-left-radius: 5px; border-bottom-left-radius: 5px; }
            QLabel#statItemValue { background: #1d2229; color: #ffffff; font-weight: 600; padding: 0 6px; border-top-right-radius: 5px; border-bottom-right-radius: 5px; }
            QFrame#statItem[summary="true"] { background: #16343d; border-color: #287286; }
            QFrame#statItem[summary="true"] QLabel#statItemLabel { background: #1d4b57; color: #bdeef5; }
            QFrame#statItem[summary="true"] QLabel#statItemValue { background: #123039; color: #65e4f4; }
        """)
