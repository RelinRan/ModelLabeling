from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QFrame, QHBoxLayout, QLabel, QLayout, QPushButton, QVBoxLayout


CONTENT_MARGIN = 14
LABEL_VALUE_SPACING = 10
ROW_SPACING = 12
BUTTON_TOP_SPACING = 16
BUTTON_SPACING = 8
BUTTON_WIDTH = 46
BUTTON_HEIGHT = 30
BUTTON_PADDING = 12


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
    """Dialog buttons size to their text with 12px horizontal padding."""
    for button in buttons:
        button.setFixedHeight(BUTTON_HEIGHT)
        width = button.fontMetrics().horizontalAdvance(button.text()) + 2 * BUTTON_PADDING
        button.setFixedWidth(max(BUTTON_WIDTH, width))


def set_confirm_button(button: QPushButton) -> None:
    """Make Enter activate the dialog's confirmation action."""
    button.setDefault(True)
    button.setAutoDefault(True)


def set_content_margins(layout: QLayout) -> None:
    layout.setContentsMargins(CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN)


def section_card(parent_layout, title: str, variant: str | None = None, badge: QWidget | None = None, badge_after_title: bool = False) -> QVBoxLayout:
    """Add a rounded section card and return its content layout.

    The card draws its own background, border, and accent-bar title, so
    sibling blocks read as clearly separated panels. By default the badge
    sits at the far right of the title row; with badge_after_title it is
    placed right after the title text.
    """
    frame = QFrame()
    frame.setObjectName("sectionCard")
    if variant:
        frame.setProperty("variant", variant)
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(12, 9, 12, 11)
    outer.setSpacing(7)
    header_row = QHBoxLayout()
    header_row.setSpacing(7)
    dot = QLabel()
    dot.setObjectName("sectionCardDot")
    dot.setFixedSize(8, 8)
    header = QLabel(title)
    header.setObjectName("sectionCardTitle")
    header_row.addWidget(dot, 0, Qt.AlignmentFlag.AlignVCenter)
    header_row.addWidget(header, 0, Qt.AlignmentFlag.AlignVCenter)
    if badge is not None and badge_after_title:
        header_row.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)
    header_row.addStretch(1)
    if badge is not None and not badge_after_title:
        header_row.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)
    outer.addLayout(header_row)
    content = QVBoxLayout()
    content.setSpacing(6)
    outer.addLayout(content)
    parent_layout.addWidget(frame)
    return content
