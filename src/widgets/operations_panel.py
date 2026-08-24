from __future__ import annotations

from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget


class OperationsPanel(QWidget):
    """Compact right-side progress panel; navigation is available from View and shortcuts."""

    def __init__(self, language: str = "zh_CN", parent=None) -> None:
        super().__init__(parent)
        self.language = language
        self.title_label = QLabel("标注操作")
        self.title_label.setObjectName("panelTitle")
        self.progress_heading = QLabel("标注进度")
        self.progress_heading.setObjectName("panelTitle")
        self.progress_text = QLabel("0% = 0/0")
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setTextVisible(False)
        self.file_label = QLabel(""); self.file_label.setObjectName("statusMuted")
        layout = QVBoxLayout(self); layout.setContentsMargins(12, 12, 12, 12); layout.setSpacing(8)
        layout.addWidget(self.title_label); self.title_label.hide(); layout.addWidget(self.progress_heading); layout.addWidget(self.progress_text); layout.addWidget(self.progress); layout.addWidget(self.file_label); layout.addStretch()
        self.navigation_toolbar = QWidget(self); self.navigation_toolbar.hide()
        # Keep legacy references parented to the hidden toolbar. Creating
        # orphan QWidget instances here can produce native blank windows on
        # Windows when the platform repaints the preview during dataset load.
        self.previous_button = QWidget(self.navigation_toolbar)
        self.next_button = QWidget(self.navigation_toolbar)
        self.zoom_in_button = QWidget(self.navigation_toolbar)
        self.zoom_out_button = QWidget(self.navigation_toolbar)
        self.fit_button = QWidget(self.navigation_toolbar)
        for button in (self.previous_button, self.next_button, self.zoom_in_button, self.zoom_out_button, self.fit_button):
            button.hide()

    def set_language(self, language: str) -> None:
        self.language = language
        self.title_label.setText("Annotation Operations" if language == "en_US" else "\u6807\u6ce8\u64cd\u4f5c")
        self.progress_heading.setText("Labeling progress" if language == "en_US" else "\u6807\u6ce8\u8fdb\u5ea6")

    def update_state(self, filename: str | None, labeled: int, total: int, labels: list[str] | None = None) -> None:
        percent = int(labeled / total * 100) if total else 0
        self.progress.setValue(percent); self.progress_text.setText(f"{percent}% = {labeled}/{total}"); self.file_label.setText(filename or "")
