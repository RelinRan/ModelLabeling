from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QScrollArea, QSizePolicy, QStyle, QVBoxLayout, QWidget

from .form_layout import section_card
from .i18n import text


class _KpiTile(QFrame):
    """A summary number: big value on top, muted caption below."""

    def __init__(self, caption: str, value: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("kpiTile")
        self.setFixedSize(92, 92)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 18, 6, 16)
        layout.setSpacing(2)
        value_view = QLabel(value)
        value_view.setObjectName("kpiValue")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        value_view.setFont(font)
        value_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        caption_view = QLabel(caption)
        caption_view.setObjectName("kpiCaption")
        caption_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value_view, 1)
        layout.addWidget(caption_view)


class _LabelPill(QFrame):
    """One label and its count; transparent, stretching to its grid cell."""

    def __init__(self, label: str, value: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("labelPill")
        self.setMinimumHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        label_view = QLabel(str(label))
        label_view.setObjectName("labelPillName")
        label_view.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        value_view = QLabel(str(value))
        value_view.setObjectName("labelPillCount")
        value_view.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label_view, 1)
        layout.addWidget(value_view)


class _TitleBadge(QLabel):
    """A value pinned to the far right of a section card's title row."""

    def __init__(self, value: str, parent=None) -> None:
        super().__init__(value, parent)
        self.setObjectName("titleBadge")


class StatsDialog(QDialog):
    def __init__(self, stats: dict, language: str = "zh_CN", parent=None) -> None:
        super().__init__(parent)
        self.language = language
        self.setObjectName("statisticsDialog")
        self.setWindowTitle(text("statistics", language))
        chinese = language != "en_US"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # ---- overview card: percent shown on the title row --------------
        overview = section_card(layout, "标注总览" if chinese else "Annotation Overview")
        overview.addSpacing(13)  # +7 layout gap = 20px below the title row

        tiles = QWidget()
        tiles_layout = QHBoxLayout(tiles)
        tiles_layout.setContentsMargins(0, 0, 0, 0)
        tiles_layout.setSpacing(10)
        captions = ("图片总数", "标注图片", "标注进度", "标签总数", "标签类别") if chinese else ("Images", "Annotated", "Progress", "Labels", "Classes")
        values = (stats["total_images"], stats["labeled_images"], f"{stats['percentage']:.1f}%", stats["total_labels"], len(stats["label_counts"]))
        for caption, value in zip(captions, values):
            tiles_layout.addWidget(_KpiTile(caption, str(value)), 1)
        overview.addWidget(tiles)
        overview.addSpacing(9)  # +11px card bottom margin = 20px

        # ---- label distribution card -----------------------------------
        counts = list(stats["label_counts"].items())
        distribution_host = QWidget()
        distribution_host.setObjectName("statsGridHost")
        grid = QGridLayout(distribution_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(9)
        grid.setVerticalSpacing(9)
        grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        for column in range(4):
            grid.setColumnStretch(column, 1)
        if counts:
            columns = 1 if len(counts) <= 1 else 2 if len(counts) <= 4 else 3 if len(counts) <= 9 else 4
            for index, (name, count) in enumerate(counts):
                row, column = divmod(index, columns)
                grid.addWidget(_LabelPill(str(name), str(count)), row, column)
        else:
            empty = QLabel("暂无标签，开始标注后这里会显示分布" if chinese else "No labels yet; the distribution appears after annotating")
            empty.setObjectName("emptyHint")
            empty.setWordWrap(True)
            grid.addWidget(empty, 0, 0)

        scroll = QScrollArea()
        self.scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(distribution_host)
        distribution_card = section_card(layout, "标签分布" if chinese else "Label distribution")
        distribution_card.addSpacing(8)  # +7 layout gap = 15px below the title row
        distribution_card.addWidget(scroll)

        # ---- sizing ------------------------------------------------------
        rows = max(1, (max(1, len(counts)) + 3) // 4)
        content_height = rows * 30 + max(0, rows - 1) * grid.verticalSpacing()
        distribution_host.setMinimumSize(0, content_height)

        screen = self.screen() or QApplication.primaryScreen()
        available_height = screen.availableGeometry().height() if screen else 900
        maximum_height = max(260, int(available_height * 0.8))
        overview_height = 190
        needs_scroll = overview_height + content_height + 90 > maximum_height
        scrollbar_width = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent) if needs_scroll else 0
        desired_width = 640 + scrollbar_width
        desired_height = min(overview_height + content_height + 70, maximum_height)
        self.resize(min(desired_width, int((screen.availableGeometry().width() if screen else 1200) * 0.9)), desired_height)

        self.setStyleSheet("""
            QLabel#titleBadge { color: #7FA3E0; font-size: 13px; font-weight: 700; border: none; background: transparent; }

            QFrame#kpiTile { background: #2F3237; border: 1px solid #3C4148; border-radius: 46px; }
            QLabel#kpiValue { color: #FFFFFF; font-size: 16px; border: none; background: transparent; }
            QLabel#kpiCaption { color: #8B94A3; font-size: 11px; border: none; background: transparent; }

            QFrame#labelPill { background: #2F3237; border: 1px solid #3E424A; border-radius: 6px; }
            QWidget#statsGridHost, QScrollArea { background: transparent; border: none; }
            QLabel#labelPillName { color: #C6CBD3; font-size: 12px; padding: 0 10px 0 11px; border: none; background: transparent; }
            QLabel#labelPillCount { color: #FFFFFF; font-size: 13px; font-weight: 700; padding: 0 10px 0 0; border: none; background: transparent; }
            QLabel#emptyHint { color: #8B9099; font-size: 12px; border: none; background: transparent; padding: 6px 2px; }
        """)
