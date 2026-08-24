from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLayout, QPushButton


CONTENT_MARGIN = 20
LABEL_VALUE_SPACING = 12
ROW_SPACING = 20
BUTTON_TOP_SPACING = 30
BUTTON_SPACING = 10
BUTTON_WIDTH = 46
BUTTON_HEIGHT = 30


def configure_form(layout: QFormLayout) -> QFormLayout:
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setHorizontalSpacing(LABEL_VALUE_SPACING)
    layout.setVerticalSpacing(ROW_SPACING)
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return layout


def configure_buttons(layout: QHBoxLayout) -> QHBoxLayout:
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(BUTTON_SPACING)
    return layout


def size_buttons(*buttons: QPushButton) -> None:
    for button in buttons:
        button.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)


def set_confirm_button(button: QPushButton) -> None:
    """Make Enter activate the dialog's confirmation action."""
    button.setDefault(True)
    button.setAutoDefault(True)


def set_content_margins(layout: QLayout) -> None:
    layout.setContentsMargins(CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN)
