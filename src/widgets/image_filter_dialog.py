from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDialog, QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget
from .form_layout import BUTTON_TOP_SPACING, configure_buttons, configure_form, set_confirm_button, set_content_margins, size_buttons


class ImageFilterDialog(QDialog):
    def __init__(self, query: str = "", status: str = "all", label: str = "", parent=None) -> None:
        super().__init__(parent)
        english = getattr(getattr(parent, "settings", None), "language", "zh_CN") == "en_US"
        self.setWindowTitle("File Filter" if english else "文件筛选"); self.setFixedWidth(360)
        self.query = QLineEdit(query); self.query.setPlaceholderText("Fuzzy file name" if english else "模糊文件名")
        self.status = QComboBox(); self.status.addItem("Labeled" if english else "已标", "labeled"); self.status.addItem("Unlabeled" if english else "未标", "unlabeled"); self.status.addItem("All" if english else "全部", "all"); self.status.setCurrentIndex(max(0, self.status.findData(status)))
        self.label_query = QLineEdit(label); self.label_query.setPlaceholderText("Label name" if english else "标签名称")
        cancel = QPushButton("Cancel" if english else "取消"); cancel.clicked.connect(self.reject)
        confirm = QPushButton("Confirm" if english else "确认"); confirm.clicked.connect(self.accept)
        buttons_layout = configure_buttons(QHBoxLayout()); buttons_layout.addStretch(); buttons_layout.addWidget(cancel); buttons_layout.addWidget(confirm)
        size_buttons(cancel, confirm)
        set_confirm_button(confirm)
        form = configure_form(QFormLayout()); form.addRow("Annotation status" if english else "标注状态", self.status); form.addRow("File name" if english else "文件名称", self.query); form.addRow("Label name" if english else "标签名称", self.label_query)
        layout = QVBoxLayout(self); set_content_margins(layout); layout.setSpacing(0); layout.addLayout(form); layout.addSpacing(BUTTON_TOP_SPACING); layout.addLayout(buttons_layout)
    def values(self) -> tuple[str, str, str]: return self.query.text().strip(), str(self.status.currentData()), self.label_query.text().strip()
