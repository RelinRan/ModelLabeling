from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QComboBox, QDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout, QWidget
from src.models.project import KeypointGroup
from .common_dialogs import AppDialog, NameDialog, dialog_language
from .form_layout import BUTTON_TOP_SPACING, configure_buttons, configure_form, set_confirm_button, set_content_margins, size_buttons


CARD_STYLE = (
    "QPushButton { background: #34373C; color: #E6E9ED; border: 1px solid #45494F; "
    "border-left: 3px solid #45494F; border-radius: 5px; padding: 2px 6px; "
    "font-weight: 600; text-align: left; padding-left: 9px; padding-right: 26px; } "
    "QPushButton:hover { background: #3C4046; border-left: 3px solid #6A84B8; } "
    'QPushButton[selected="true"] { background: #31436B; border: 1px solid #6A84B8; '
    "border-left: 3px solid #7FA3E0; color: #FFFFFF; }"
)


class KeypointTypeDialog(QDialog):
    """Name, count, and label inputs for creating or editing a keypoint type.

    The label is the annotation label used when a drawing made with this type
    completes; the combo offers the existing label presets and free text.
    """

    def __init__(self, title: str, name: str = "", count: int = 1, label: str = "",
                 labels: list[str] | None = None, parent=None, language: str | None = None) -> None:
        super().__init__(parent)
        english = dialog_language(parent, language) == "en_US"
        self.setWindowTitle(title)
        self.name_edit = QLineEdit(name)
        self.count_box = QSpinBox(); self.count_box.setRange(1, 135); self.count_box.setValue(max(1, int(count)))
        self.label_box = QComboBox(); self.label_box.setEditable(True)
        self.label_box.addItems([item for item in (labels or []) if item])
        self.label_box.setCurrentText(label or (""))
        form = configure_form(QFormLayout())
        form.addRow("Name" if english else "名称", self.name_edit)
        form.addRow("Keypoint count" if english else "点数", self.count_box)
        form.addRow("Label" if english else "标签", self.label_box)
        layout = QVBoxLayout(self); set_content_margins(layout); layout.setSpacing(0); layout.addLayout(form); layout.addSpacing(BUTTON_TOP_SPACING)
        buttons = configure_buttons(QHBoxLayout())
        cancel = QPushButton("Cancel" if english else "取消"); cancel.clicked.connect(self.reject)
        confirm = QPushButton("Confirm" if english else "确认"); confirm.clicked.connect(self.accept)
        buttons.addStretch(); buttons.addWidget(cancel); buttons.addWidget(confirm); layout.addLayout(buttons)
        size_buttons(cancel, confirm)
        set_confirm_button(confirm)

    def label_value(self) -> str:
        return self.label_box.currentText().strip()


class KeypointGroupsDialog(QDialog):
    """Edit keypoint types: type list on the left, point-name cards on the right.

    The right pane mirrors the label-groups panel's card grid (same hover,
    selection, and focus styling) so both editors feel identical.
    """

    GRID_SPACING = 7
    LEFT_PADDING = 9
    RIGHT_PADDING = 26

    def __init__(self, groups: list[KeypointGroup], parent=None, language: str = "zh_CN", labels: list[str] | None = None) -> None:
        super().__init__(parent)
        self.language = language; english = language == "en_US"
        self.preset_labels = [str(item) for item in (labels or []) if str(item).strip()]
        self.setWindowTitle("Keypoint Types" if english else "\u70b9\u4f4d\u7c7b\u578b"); self.setMinimumSize(760, 500)
        self.groups = [
            KeypointGroup(group.name, list(group.keypoint_names), group.protected, group.label)
            for group in groups
        ]
        self._selected_index = -1
        self._grid_columns = 1
        self.group_list = _TypeList()
        self.group_list.currentRowChanged.connect(self._select_group)
        self.add_group_button = QPushButton("Add Type" if english else "\u65b0\u589e\u7c7b\u578b")
        self.edit_group_button = QPushButton("Edit Type" if english else "\u7f16\u8f91\u7c7b\u578b")
        self.delete_group_button = QPushButton("Delete Type" if english else "\u5220\u9664\u7c7b\u578b")
        for button, handler in ((self.add_group_button, self.add_group), (self.edit_group_button, self.edit_group), (self.delete_group_button, self.delete_group)): button.clicked.connect(handler)
        self.add_name_button = QPushButton("Add" if english else "\u65b0\u589e")
        self.edit_name_button = QPushButton("Edit" if english else "\u7f16\u8f91")
        self.delete_name_button = QPushButton("Delete" if english else "\u5220\u9664")
        for button, handler in ((self.add_name_button, self.add_name), (self.edit_name_button, self.edit_name), (self.delete_name_button, self.delete_name)): button.clicked.connect(handler)
        self.confirm_button = QPushButton("Confirm" if english else "\u786e\u8ba4")
        self.confirm_button.clicked.connect(self.accept)
        set_confirm_button(self.confirm_button)

        left = QFrame(); left.setObjectName("labelArea")
        left_layout = QVBoxLayout(left); left_layout.setContentsMargins(8, 8, 8, 8); left_layout.setSpacing(8); left_layout.addWidget(self.group_list)
        group_actions = QHBoxLayout(); group_actions.addWidget(self.add_group_button); group_actions.addWidget(self.edit_group_button); group_actions.addWidget(self.delete_group_button); left_layout.addLayout(group_actions)

        # Right pane: card grid identical in structure to PresetPanel's.
        self.card_frame = QFrame(); self.card_frame.setObjectName("labelArea")
        frame_layout = QVBoxLayout(self.card_frame); frame_layout.setContentsMargins(8, 8, 8, 8)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host); self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(self.GRID_SPACING); self.grid.setVerticalSpacing(self.GRID_SPACING)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.card_scroll = QScrollArea(); self.card_scroll.setWidgetResizable(True)
        self.card_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.card_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.card_scroll.setWidget(self.grid_host)
        frame_layout.addWidget(self.card_scroll)
        for widget in (self.card_frame, self.grid_host, self.card_scroll, self.card_scroll.viewport()):
            widget.setStyleSheet("background-color: #25272A; border: none;")
            widget.setAutoFillBackground(True)
        for widget in (self.card_frame, self.grid_host, self.card_scroll.viewport()): widget.installEventFilter(self)

        right = QFrame(); right.setObjectName("labelGroupsRight")
        right.setStyleSheet("QFrame#labelGroupsRight { background: #25272A; border: 1px solid #464A50; border-radius: 5px; }")
        right_layout = QVBoxLayout(right); right_layout.setContentsMargins(8, 8, 8, 8); right_layout.setSpacing(8)
        right_layout.addWidget(self.card_frame, 1)
        # Same single action row as the label-groups panel: point-name actions
        # plus the dialog confirm, all on one line.
        name_actions = QHBoxLayout(); name_actions.setContentsMargins(0, 0, 0, 0); name_actions.setSpacing(6)
        for button in (self.add_name_button, self.edit_name_button, self.delete_name_button): name_actions.addWidget(button)
        name_actions.addWidget(self.confirm_button)
        right_layout.addLayout(name_actions)

        body = QHBoxLayout(); body.setSpacing(14); body.addWidget(left, 1); body.addWidget(right, 2)
        layout = QVBoxLayout(self); layout.setContentsMargins(18, 18, 18, 18); layout.addLayout(body)
        self._refresh_groups(); self.group_list.setCurrentRow(0)

    @property
    def english(self) -> bool: return self.language == "en_US"

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.MouseButtonDblClick and watched.property("point_index") is not None:
            self._selected_index = int(watched.property("point_index"))
            self._apply_selection()
            self.edit_name()
            return True
        if watched in {self.card_frame, self.grid_host, self.card_scroll.viewport()} and event.type() == QEvent.Type.MouseButtonPress:
            self._select_index(-1)
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        group = self._group()
        if group is not None and getattr(self, "_last_seen_columns", None) != self._preferred_columns(group.keypoint_names):
            self._render_names()

    # ---- left: types ----

    def _refresh_groups(self) -> None:
        # Rebuilding the list resets the selection to row 0; keep the edited
        # row selected so the right pane does not jump to another type.
        row = self.group_list.currentRow()
        self.group_list.clear()
        for group in self.groups:
            parts = [group.name]
            if group.keypoint_names:
                parts.append(f"({len(group.keypoint_names)}{' pts' if self.english else '点'})")
            if group.label:
                parts.append(f"· {group.label}")
            self.group_list.addItem("  ".join(parts))
        if self.group_list.count():
            self.group_list.setCurrentRow(min(max(row, 0), self.group_list.count() - 1))

    def _select_group(self, row: int) -> None:
        self._selected_index = -1
        self._render_names()

    def _group(self) -> KeypointGroup | None:
        row = self.group_list.currentRow()
        return self.groups[row] if 0 <= row < len(self.groups) else None

    def _resize_names(self, group: KeypointGroup, count: int) -> None:
        """Grow or shrink a type's point-name list to the new count.

        Added slots get generated names; shrinking drops the tail. Existing
        custom names are preserved so an edit never silently renames points.
        """
        count = max(1, min(135, int(count)))
        while len(group.keypoint_names) > count:
            group.keypoint_names.pop()
        index = len(group.keypoint_names) + 1
        while len(group.keypoint_names) < count:
            name = f"kpt_{index}"
            while name in group.keypoint_names:
                index += 1
                name = f"kpt_{index}"
            group.keypoint_names.append(name)
            index += 1

    def add_group(self) -> None:
        dialog = KeypointTypeDialog("Add Keypoint Type" if self.english else "\u65b0\u589e\u70b9\u4f4d\u7c7b\u578b", labels=self.preset_labels, parent=self, language=self.language)
        if dialog.exec() == dialog.DialogCode.Accepted:
            name = dialog.name_edit.text().strip()
            if not name or any(group.name == name for group in self.groups):
                return
            group = KeypointGroup(name, label=dialog.label_value())
            self._resize_names(group, dialog.count_box.value())
            self.groups.append(group); self._refresh_groups(); self.group_list.setCurrentRow(len(self.groups) - 1)

    def edit_group(self) -> None:
        row = self.group_list.currentRow()
        if row < 0: return
        group = self.groups[row]
        dialog = KeypointTypeDialog("Edit Keypoint Type" if self.english else "\u7f16\u8f91\u70b9\u4f4d\u7c7b\u578b", group.name, len(group.keypoint_names), group.label, self.preset_labels, self, self.language)
        if dialog.exec() == dialog.DialogCode.Accepted:
            name = dialog.name_edit.text().strip()
            if not name: return
            group.name = name
            group.label = dialog.label_value()
            self._resize_names(group, dialog.count_box.value())
            self._refresh_groups(); self.group_list.setCurrentRow(row)

    def delete_group(self) -> None:
        row = self.group_list.currentRow()
        if row < 0: return
        if self.groups[row].protected:
            AppDialog.information("提示" if self.english else "\u70b9\u4f4d\u7c7b\u578b", "The default keypoint type cannot be deleted." if self.english else "\u9ed8\u8ba4\u70b9\u4f4d\u7c7b\u578b\u4e0d\u53ef\u5220\u9664\u3002", self); return
        name = self.groups[row].name
        if not AppDialog.question("提示" if self.english else "\u5220\u9664\u70b9\u4f4d\u7c7b\u578b", f"Delete keypoint type '{name}'?" if self.english else f"\u786e\u8ba4\u5220\u9664\u70b9\u4f4d\u7c7b\u578b\u201c{name}\u201d\u5417\uff1f", self): return
        self.groups.pop(row); self._refresh_groups(); self.group_list.setCurrentRow(max(0, row - 1))

    # ---- right: point-name cards ----

    def _preferred_columns(self, names: list[str]) -> int:
        """Pick 3/2/1 columns so every card's content fits without clipping."""
        available = max(120, self.card_scroll.viewport().width() or 284)
        metrics = QFontMetrics(self.font())
        widest = max((metrics.horizontalAdvance(name) for name in names), default=0)
        needed = widest + self.LEFT_PADDING + self.RIGHT_PADDING + 2  # + border
        for columns in (3, 2, 1):
            if columns * needed + (columns - 1) * self.GRID_SPACING <= available:
                return columns
        return 1

    def _render_names(self) -> None:
        group = self._group()
        names = group.keypoint_names if group else []
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        columns = self._preferred_columns(names)
        self._grid_columns = columns
        self._last_seen_columns = columns
        metrics = QFontMetrics(self.font())
        available = max(120, self.card_scroll.viewport().width() or 284)
        card_width = (available - (columns - 1) * self.GRID_SPACING) // max(1, columns)
        text_width = card_width - self.LEFT_PADDING - self.RIGHT_PADDING - 2
        for index, name in enumerate(names):
            label_text = name
            if metrics.horizontalAdvance(label_text) > text_width:
                label_text = metrics.elidedText(label_text, Qt.TextElideMode.ElideRight, max(20, text_width))
            card = QPushButton(label_text)
            chip = QLabel(f"{index + 1}", card)
            chip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            chip.setStyleSheet("color: #8B9099; font-size: 11px; font-weight: 400; border: none; background: transparent;")
            chip_row = QHBoxLayout(card); chip_row.setContentsMargins(0, 0, 9, 0); chip_row.setSpacing(0)
            chip_row.addStretch(1); chip_row.addWidget(chip)
            card.setObjectName("labelCard"); card.setFixedHeight(36); card.setMinimumWidth(0)
            card.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card.setProperty("selected", False); card.setProperty("point_index", index)
            card.setStyleSheet(CARD_STYLE)
            card.clicked.connect(lambda checked=False, row=index: self._select_index(row))
            card.installEventFilter(self)
            row, column = divmod(index, columns)
            self.grid.addWidget(card, row, column)
        if self._selected_index >= len(names):
            self._selected_index = -1
        self._apply_selection()
        self.grid_host.adjustSize()

    def _apply_selection(self) -> None:
        for index in range(self.grid.count()):
            widget = self.grid.itemAt(index).widget()
            if widget is None: continue
            selected = index == self._selected_index
            widget.setProperty("selected", selected)
            widget.style().unpolish(widget); widget.style().polish(widget)

    def _select_index(self, index: int) -> None:
        self._selected_index = index
        self._apply_selection()

    def add_name(self) -> None:
        group = self._group()
        if group is None: return
        dialog = NameDialog("Add Keypoint" if self.english else "\u65b0\u589e\u70b9\u4f4d", parent=self, language=self.language)
        if dialog.exec() == dialog.DialogCode.Accepted:
            name = dialog.name_edit.text().strip()
            if name and name not in group.keypoint_names:
                group.keypoint_names.append(name)
                self._refresh_groups(); self._render_names(); self._select_index(len(group.keypoint_names) - 1)

    def edit_name(self) -> None:
        group = self._group()
        if group is None or not (0 <= self._selected_index < len(group.keypoint_names)): return
        row = self._selected_index
        dialog = NameDialog("Edit Keypoint" if self.english else "\u7f16\u8f91\u70b9\u4f4d", group.keypoint_names[row], self, self.language)
        if dialog.exec() == dialog.DialogCode.Accepted:
            name = dialog.name_edit.text().strip()
            if name and name not in group.keypoint_names:
                group.keypoint_names[row] = name
                self._refresh_groups(); self._render_names(); self._select_index(row)

    def delete_name(self) -> None:
        group = self._group()
        if group is None or not (0 <= self._selected_index < len(group.keypoint_names)): return
        row = self._selected_index
        group.keypoint_names.pop(row)
        self._refresh_groups(); self._render_names(); self._select_index(max(0, row - 1))


class _TypeList(QListWidget):
    """Left type list sharing the label-groups dialog's list styling."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("groupFileList")
        self.setStyleSheet(
            "QListWidget#groupFileList, QListWidget#groupFileList:focus { background: #25272A; border: 1px solid #464A50; border-radius: 5px; padding: 6px; outline: 0; } "
            "QListWidget#groupFileList::item { height: 30px; padding: 0 5px; margin: 0; background: #35383D; color: #FFFFFF; border: 2px solid transparent; border-radius: 5px; margin-bottom: 6px; } "
            "QListWidget#groupFileList::item:hover { background: #41454C; color: #FFFFFF; border: 2px solid #FFFFFF; } "
            "QListWidget#groupFileList::item:selected, QListWidget#groupFileList::item:selected:focus { background: #2e436e; color: #FFFFFF; font-weight: 600; border: 2px solid #FFFFFF; outline: 0; }"
        )
