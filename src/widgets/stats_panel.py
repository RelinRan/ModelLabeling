from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget


class StatsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.summary = QLabel()
        self.labels = QLabel()
        self.summary.setWordWrap(True)
        self.labels.setWordWrap(True)
        self.summary.setObjectName("statsSummary")
        self.labels.setObjectName("statsLabels")
        self.title_label = QLabel("数据统计")
        self.title_label.setObjectName("panelTitle")
        self.summary_card = QFrame()
        self.summary_card.setObjectName("statsCard")
        summary_layout = QGridLayout(self.summary_card)
        summary_layout.setContentsMargins(12, 10, 12, 10)
        summary_layout.addWidget(self.summary, 0, 0)
        self.labels_card = QFrame()
        self.labels_card.setObjectName("statsCard")
        labels_layout = QVBoxLayout(self.labels_card)
        labels_layout.setContentsMargins(12, 10, 12, 10)
        labels_layout.addWidget(self.labels)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self.title_label)
        layout.addWidget(self.summary_card)
        layout.addWidget(self.labels_card)
        layout.addStretch()

    def update_stats(self, stats: dict) -> None:
        self.summary.setText(
            f"图片总数  {stats['total_images']}    "
            f"已标注  {stats['labeled_images']}\n"
            f"完成度  {stats['percentage']:.1f}%    "
            f"标签总数  {stats['total_labels']}"
        )
        counts = stats["label_counts"]
        self.labels.setText(
            "\n".join(f"{name:<14} {count:>4}" for name, count in counts.items()) or "暂无标签"
        )
