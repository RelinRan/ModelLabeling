from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDialog, QFormLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from .common_dialogs import choose_color
from .form_layout import BUTTON_TOP_SPACING, ROW_SPACING, configure_buttons, configure_form, set_confirm_button, set_content_margins, size_buttons
from .numeric_stepper import NumericStepper


class CrosshairDialog(QDialog):
    changed = Signal(int, str)

    def __init__(self, line_width: int, color: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("标注辅助")
        self.setMinimumWidth(320)
        self.line_width = NumericStepper(line_width, 1, 12, 1)
        self.color_button = QPushButton("选择颜色")
        self.color_button.setFixedHeight(30)
        self._color = QColor(color)
        self._refresh_color_button()
        self.color_button.clicked.connect(self._choose_color)
        form = configure_form(QFormLayout())
        form.addRow("线条粗细", self.line_width)
        form.addRow("线条颜色", self.color_button)
        layout = QVBoxLayout(self)
        set_content_margins(layout)
        layout.setSpacing(0)
        layout.addWidget(QLabel("注:辅助线会在 W 绘制模式下显示"))
        layout.addSpacing(ROW_SPACING)
        layout.addLayout(form)
        layout.addSpacing(BUTTON_TOP_SPACING)
        buttons = QWidget()
        buttons_layout = configure_buttons(QHBoxLayout(buttons))
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        confirm = QPushButton("确认")
        confirm.clicked.connect(self._accept_changes)
        buttons_layout.addStretch()
        buttons_layout.addWidget(cancel)
        buttons_layout.addWidget(confirm)
        size_buttons(cancel, confirm)
        set_confirm_button(confirm)
        layout.addWidget(buttons)

    def _refresh_color_button(self) -> None:
        self.color_button.setStyleSheet(
            f"QPushButton {{ background: {self._color.name()}; color: white; }}"
        )

    def _emit_changed(self) -> None:
        self.changed.emit(self.line_width.value(), self._color.name())

    def _accept_changes(self) -> None:
        self._emit_changed()
        self.accept()

    def _choose_color(self) -> None:
        color = choose_color(self, self._color, "选择辅助线颜色")
        if color.isValid():
            self._color = color
            self._refresh_color_button()
