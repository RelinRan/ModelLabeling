from __future__ import annotations

from PySide6.QtWidgets import QDialog, QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout
from .numeric_stepper import NumericStepper
from .form_layout import BUTTON_TOP_SPACING, configure_buttons, configure_form, set_confirm_button, set_content_margins, size_buttons


class LabelEditDialog(QDialog):
    def __init__(self, group_name: str, label_name: str, class_id: int = 0, parent=None, language: str = "zh_CN") -> None:
        super().__init__(parent)
        english = language == "en_US"
        title = "Label Editor" if english else "\u6807\u7b7e\u7f16\u8f91"
        self.setWindowTitle(f"{group_name} - {title}")
        self.setFixedWidth(300)
        self.name_edit = QLineEdit(label_name)
        self.class_id = NumericStepper(class_id, 0, 9999, 1)
        confirm = QPushButton("Confirm" if english else "\u786e\u8ba4"); cancel = QPushButton("Cancel" if english else "\u53d6\u6d88")
        confirm.clicked.connect(self.accept); cancel.clicked.connect(self.reject)
        form = configure_form(QFormLayout()); form.addRow("Name" if english else "\u540d\u79f0", self.name_edit); form.addRow("Class ID" if english else "\u7c7b\u522b ID", self.class_id)
        buttons = configure_buttons(QHBoxLayout()); buttons.addStretch(); buttons.addWidget(cancel); buttons.addWidget(confirm)
        size_buttons(cancel, confirm)
        set_confirm_button(confirm)
        layout = QVBoxLayout(self); set_content_margins(layout); layout.setSpacing(0); layout.addLayout(form); layout.addSpacing(BUTTON_TOP_SPACING); layout.addLayout(buttons)
        self.adjustSize()
