from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout
from PySide6.QtWidgets import QFormLayout
from .form_layout import BUTTON_TOP_SPACING, configure_buttons, configure_form, set_confirm_button, set_content_margins, size_buttons


def dialog_language(parent=None, language: str | None = None) -> str:
    if language:
        return language
    window = parent.window() if parent is not None else None
    return getattr(getattr(window, "settings", None), "language", "zh_CN")


class AppDialog(QDialog):
    """Small icon-free application dialog used for messages and confirmations."""

    def __init__(self, title: str, message: str, confirm: bool = False, parent=None, language: str | None = None) -> None:
        super().__init__(parent)
        # Keep the public message dialog's surface consistent with the app.
        # This is scoped to QDialog and does not recolor any main-window panel.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(palette.ColorRole.Window, QColor("#2B2D30"))
        self.setPalette(palette)
        self.setStyleSheet(
            "QDialog { background-color: #2B2D30; color: #D7DAE0; }"
            "QLabel { color: #D7DAE0; background: transparent; }"
            "QPushButton { background: #35383D; color: #D7DAE0; "
            "border: 1px solid #464A50; border-radius: 5px; padding: 5px 10px; }"
            "QPushButton:hover { background: #41454C; color: #FFFFFF; }"
        )
        english = dialog_language(parent, language) == "en_US"
        self.setWindowTitle(title); self.setMinimumWidth(360); self.confirmed = False
        layout = QVBoxLayout(self); layout.setContentsMargins(22, 20, 22, 18); layout.setSpacing(18)
        message_label = QLabel(message); message_label.setWordWrap(True); message_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter); layout.addWidget(message_label)
        buttons = QHBoxLayout(); buttons.setContentsMargins(0, 12, 0, 0); buttons.addStretch()
        if confirm:
            cancel = QPushButton("Cancel" if english else "\u53d6\u6d88"); cancel.clicked.connect(self.reject); buttons.addWidget(cancel)
            accept = QPushButton("Confirm" if english else "\u786e\u8ba4"); accept.clicked.connect(self._confirm); buttons.addWidget(accept)
        else:
            ok = QPushButton("Confirm" if english else "\u786e\u8ba4"); ok.clicked.connect(self.accept); buttons.addWidget(ok)
        layout.addLayout(buttons)

    def _confirm(self) -> None:
        self.confirmed = True; self.accept()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        parent = self.parentWidget()
        if parent is not None:
            self.setWindowModality(Qt.WindowModality.ApplicationModal)
            frame = parent.window().frameGeometry()
            target = frame.center() - self.rect().center()
            self.move(target)

    @classmethod
    def information(cls, title: str, message: str, parent=None, language: str | None = None) -> None:
        cls(title, message, False, parent, language).exec()

    @classmethod
    def question(cls, title: str, message: str, parent=None, language: str | None = None) -> bool:
        if parent is not None and not parent.isVisible(): return True
        dialog = cls(title, message, True, parent, language); dialog.exec(); return dialog.confirmed

    @classmethod
    def save_choice(cls, title: str, message: str, parent=None, language: str | None = None) -> str:
        if parent is not None and not parent.isVisible(): return "discard"
        english = dialog_language(parent, language) == "en_US"
        dialog = cls.__new__(cls); QDialog.__init__(dialog, parent); dialog.setWindowTitle(title); dialog.setMinimumWidth(380)
        layout = QVBoxLayout(dialog); layout.setContentsMargins(22, 20, 22, 18)
        message_label = QLabel(message); message_label.setWordWrap(True); layout.addWidget(message_label)
        buttons = QHBoxLayout(); buttons.setContentsMargins(0, 12, 0, 0); buttons.addStretch(); result = {"value": "cancel"}
        labels = (("Cancel" if english else "\u53d6\u6d88", "cancel"), ("Discard" if english else "\u653e\u5f03", "discard"), ("Save" if english else "\u4fdd\u5b58", "save"))
        for text, value in labels:
            button = QPushButton(text); button.clicked.connect(lambda checked=False, value=value: (result.__setitem__("value", value), dialog.accept())); buttons.addWidget(button)
        layout.addLayout(buttons); dialog.exec(); return result["value"]


class NameDialog(QDialog):
    def __init__(self, title: str, value: str = "", parent=None, language: str | None = None) -> None:
        super().__init__(parent)
        english = dialog_language(parent, language) == "en_US"
        self.setWindowTitle(title); self.name_edit = QLineEdit(value)
        form = configure_form(QFormLayout()); form.addRow("Name" if english else "\u540d\u79f0", self.name_edit)
        layout = QVBoxLayout(self); set_content_margins(layout); layout.setSpacing(0); layout.addLayout(form); layout.addSpacing(BUTTON_TOP_SPACING)
        buttons = configure_buttons(QHBoxLayout())
        cancel = QPushButton("Cancel" if english else "\u53d6\u6d88"); cancel.clicked.connect(self.reject)
        confirm = QPushButton("Confirm" if english else "\u786e\u8ba4"); confirm.clicked.connect(self.accept)
        buttons.addStretch(); buttons.addWidget(cancel); buttons.addWidget(confirm); layout.addLayout(buttons)
        size_buttons(cancel, confirm)
        set_confirm_button(confirm)


def choose_color(parent, color: QColor | str, title: str | None = None, language: str | None = None) -> QColor:
    english = dialog_language(parent, language) == "en_US"
    dialog = QColorDialog(QColor(color), parent)
    dialog.setWindowTitle(title or ("Choose Label Color" if english else "\u9009\u62e9\u6807\u7b7e\u989c\u8272")); dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
    if not english:
        translations = {"Basic colors": "\u57fa\u672c\u989c\u8272", "Custom colors": "\u81ea\u5b9a\u4e49\u989c\u8272", "Pick Screen Color": "\u62fe\u53d6\u5c4f\u5e55\u989c\u8272", "Add to Custom Colors": "\u6dfb\u52a0\u5230\u81ea\u5b9a\u4e49\u989c\u8272", "Hue:": "\u8272\u76f8\uff1a", "Sat:": "\u9971\u548c\u5ea6\uff1a", "Val:": "\u660e\u5ea6\uff1a", "Red:": "\u7ea2\uff1a", "Green:": "\u7eff\uff1a", "Blue:": "\u84dd\uff1a", "Alpha channel:": "\u900f\u660e\u5ea6\uff1a", "HTML:": "HTML\uff1a"}
        for label in dialog.findChildren(QLabel): label.setText(translations.get(label.text(), label.text()))
    button_box = dialog.findChild(QDialogButtonBox)
    if button_box:
        cancel = button_box.button(QDialogButtonBox.StandardButton.Cancel); confirm = button_box.button(QDialogButtonBox.StandardButton.Ok)
        if cancel: cancel.setText("Cancel" if english else "\u53d6\u6d88")
        if confirm: confirm.setText("Confirm" if english else "\u786e\u8ba4")
        # Keep the color picker action order consistent with the app dialogs:
        # cancel on the left, confirm on the right.
        if cancel is not None and confirm is not None:
            layout = button_box.layout()
            if layout is not None:
                layout.removeWidget(cancel)
                layout.removeWidget(confirm)
                layout.addWidget(cancel)
                layout.addWidget(confirm)
    # QColorDialog contains internal spin boxes for RGB/HSV/alpha values.
    # Keep them as direct text inputs so every numeric editor behaves alike.
    for spinbox in dialog.findChildren(QSpinBox):
        spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        spinbox.setEnabled(True)
        spinbox.setStyleSheet(
            "QSpinBox { background: #35383D; border: 1px solid #464A50; border-radius: 5px; padding: 5px; }"
        )
    return dialog.selectedColor() if dialog.exec() == dialog.DialogCode.Accepted else QColor()
