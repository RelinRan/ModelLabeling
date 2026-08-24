from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QWidget


class NumericStepper(QWidget):
    """Compact numeric editor with direct text input only."""

    valueChanged = Signal(object)

    def __init__(self, value=0, minimum=0, maximum=999999, step=1, decimals=0, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._step = float(step)
        self._decimals = max(0, int(decimals))
        self._value = float(value)
        self.setMinimumHeight(30)
        self.setObjectName("numericStepper")

        self.editor = QLineEdit()
        self.editor.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.editor.setFrame(False)
        self.editor.setObjectName("numericValue")
        self.editor.setValidator(
            QDoubleValidator(self._minimum, self._maximum, self._decimals, self.editor)
            if self._decimals else QIntValidator(int(self._minimum), int(self._maximum), self.editor)
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 0)
        layout.setSpacing(0)
        layout.addWidget(self.editor, 1)
        self.setStyleSheet(
            "QWidget#numericStepper { background: #303236; border: 1px solid #464A50; border-radius: 5px; padding: 0; }"
            "QWidget#numericStepper:focus-within { border-color: #6A84B8; }"
            "QLineEdit#numericValue { background: transparent; border: none; color: #D7DAE0; padding: 0; selection-background-color: #2E436E; }"
        )
        self.editor.editingFinished.connect(self._commit_text)
        self._set_editor_value()

    def setRange(self, minimum, maximum) -> None:
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        if self._decimals:
            self.editor.setValidator(QDoubleValidator(self._minimum, self._maximum, self._decimals, self.editor))
        else:
            self.editor.setValidator(QIntValidator(int(self._minimum), int(self._maximum), self.editor))
        self.setValue(self._value)

    def setSingleStep(self, step) -> None:
        self._step = float(step)

    def setValue(self, value) -> None:
        bounded = max(self._minimum, min(self._maximum, float(value)))
        if self._decimals:
            bounded = round(bounded, self._decimals)
        self._value = bounded
        self._set_editor_value()

    def value(self):
        return round(self._value, self._decimals) if self._decimals else int(round(self._value))

    def _set_editor_value(self) -> None:
        text = f"{self._value:.{self._decimals}f}" if self._decimals else str(int(round(self._value)))
        self.editor.setText(text)

    def _commit_text(self) -> None:
        try:
            value = float(self.editor.text())
        except ValueError:
            self._set_editor_value()
            return
        old = self.value()
        self.setValue(value)
        if self.value() != old:
            self.valueChanged.emit(self.value())
