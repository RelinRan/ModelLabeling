from __future__ import annotations

from PySide6.QtCore import QEvent, QSettings, Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget, QInputDialog

from src.models.annotation import LabelPreset
from src.models.project import LabelGroup
from .common_dialogs import AppDialog
from .label_edit_dialog import LabelEditDialog


class PresetPanel(QWidget):
    presetSelected = Signal(str)
    presetsChanged = Signal()
    groupsChanged = Signal()

    def __init__(self, parent=None, language: str = "zh_CN") -> None:
        super().__init__(parent)
        self.setObjectName("projectPanel")
        self.language = language; self.presets: list[LabelPreset] = []; self.groups: list[LabelGroup] = []
        self._selected_name: str | None = None
        self._hotkey_bindings: dict[int, str] = {}
        settings = QSettings("RelinRan", "ModelLabeling")
        for number in range(1, 10):
            label = str(settings.value(f"hotkey/{number}", "") or "")
            if label:
                self._hotkey_bindings[number] = label
        self.title_label = QLabel("\u6807\u7b7e\u8bbe\u7f6e"); self.title_label.hide()
        self.group_combo = QComboBox(); self.group_combo.currentIndexChanged.connect(self._group_selected)
        self.list = QListWidget(); self.list.hide(); self.list.currentRowChanged.connect(self._selected)
        self.label_frame = QFrame(); self.label_frame.setObjectName("labelArea")
        frame_layout = QVBoxLayout(self.label_frame); frame_layout.setContentsMargins(8, 8, 8, 8)
        self.grid_host = QWidget(); self.grid = QGridLayout(self.grid_host); self.grid.setContentsMargins(0, 0, 0, 0); self.grid.setHorizontalSpacing(7); self.grid.setVerticalSpacing(7); self.grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.label_scroll = QScrollArea(); self.label_scroll.setWidgetResizable(True); self.label_scroll.setFrameShape(QFrame.Shape.NoFrame); self.label_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); self.label_scroll.setWidget(self.grid_host); frame_layout.addWidget(self.label_scroll)
        # Keep the label surface opaque while the selected user template is repainted.
        # Without this, Qt can briefly expose the parent palette during the
        # two-step group/preset refresh and show a purple flash.
        for widget in (self.label_frame, self.grid_host, self.label_scroll, self.label_scroll.viewport()):
            widget.setStyleSheet("background-color: #25272A; border: none;")
            widget.setAutoFillBackground(True)
        for widget in (self.label_frame, self.grid_host, self.label_scroll.viewport()): widget.installEventFilter(self)
        self.button_frame = QFrame(); self.button_frame.setObjectName("labelActions")
        actions = QHBoxLayout(self.button_frame); actions.setContentsMargins(8, 8, 8, 8); actions.setSpacing(6)
        self.add_button = QPushButton(); self.edit_button = QPushButton(); self.delete_button = QPushButton()
        for button in (self.add_button, self.edit_button, self.delete_button): actions.addWidget(button)
        layout = QVBoxLayout(self); layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8); layout.addWidget(self.group_combo); layout.addWidget(self.label_frame, 1); layout.addWidget(self.button_frame)
        self.add_button.clicked.connect(self.add_preset); self.edit_button.clicked.connect(self.edit_selected); self.delete_button.clicked.connect(self.delete_selected)
        # Labels referenced by keypoint types cannot be deleted from the
        # library while a type still uses them.
        self._protected_labels: set[str] = set()
        self.set_language(language)

    def set_protected_labels(self, labels: set[str]) -> None:
        self._protected_labels = {str(name) for name in labels if name}

    def set_language(self, language: str) -> None:
        self.language = language; english = language == "en_US"
        self.add_button.setText("Add" if english else "\u65b0\u589e"); self.edit_button.setText("Edit" if english else "\u7f16\u8f91")
        self.delete_button.setText("Delete" if english else "\u5220\u9664")
        if self.groups:
            selected = self.selected_group().name if self.selected_group() else None
            self.set_groups(self.groups, selected)

    def set_presets(self, presets: list[LabelPreset]) -> None: self.presets = list(presets); self._render_presets()

    def refresh_selected_group(self) -> None:
        """Synchronously repaint the cards for the currently selected group."""
        group = self.selected_group()
        if group is not None:
            self.presets = group.presets
        self._render_presets()
        self.grid_host.adjustSize()
        self.label_frame.repaint()
        self.label_scroll.viewport().repaint()
        self.grid_host.repaint()

    def set_groups(self, groups: list[LabelGroup], selected: str | None = None) -> None:
        self.groups = groups; self.group_combo.blockSignals(True); self.group_combo.clear()
        for group in groups:
            display_name = group.name
            if group.protected and group.name in {"默认标签", "Default Labels", "Default"}:
                display_name = "Default Labels" if self.language == "en_US" else "默认标签"
            self.group_combo.addItem(display_name, group.name)
        index = self.group_combo.findData(selected) if selected else 0
        if index < 0 and selected:
            index = self.group_combo.findText(selected)
        self.group_combo.setCurrentIndex(max(0, index)); self.group_combo.blockSignals(False)
        self._update_group_combo_arrow()
        self._group_selected(self.group_combo.currentIndex())

    def _update_group_combo_arrow(self) -> None:
        """Show the dropdown affordance only when there are choices."""
        hidden = len(self.groups) <= 1
        self.group_combo.setProperty("singleGroup", hidden)
        self.group_combo.style().unpolish(self.group_combo)
        self.group_combo.style().polish(self.group_combo)
        self.group_combo.update()

    def selected_group(self) -> LabelGroup | None:
        index = self.group_combo.currentIndex(); return self.groups[index] if 0 <= index < len(self.groups) else None

    def _group_selected(self, index: int) -> None:
        if 0 <= index < len(self.groups): self.presets = self.groups[index].presets
        self._render_presets()

    def _clear_grid(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget:
                # Keep the grid host as the parent until Qt processes the
                # deferred deletion. Detaching first promotes the card to a
                # temporary top-level window at (0, 0), which flashes on
                # screen when a dialog is confirmed.
                widget.deleteLater()

    GRID_SPACING = 7
    LEFT_PADDING = 9
    RIGHT_PADDING = 26

    def _preferred_columns(self) -> int:
        """Pick 3/2/1 columns so every card's content fits without clipping."""
        available = max(120, self.label_scroll.viewport().width() or 284)
        metrics = QFontMetrics(self.font())
        widest = 0
        for index, preset in enumerate(self.presets):
            hint = self._hotkey_hint(index)
            widest = max(widest, metrics.horizontalAdvance(f"{hint}{preset.name}"))
        needed = widest + self.LEFT_PADDING + self.RIGHT_PADDING + 2  # + border
        for columns in (3, 2, 1):
            if columns * needed + (columns - 1) * self.GRID_SPACING <= available:
                return columns
        return 1

    def _render_presets(self) -> None:
        self.list.clear(); self.list.addItems([preset.name for preset in self.presets]); self._clear_grid()
        columns = self._preferred_columns()
        self.grid_columns = columns
        metrics = QFontMetrics(self.font())
        available = max(120, self.label_scroll.viewport().width() or 284)
        card_width = (available - (columns - 1) * self.GRID_SPACING) // max(1, columns)
        text_width = card_width - self.LEFT_PADDING - self.RIGHT_PADDING - 2
        for index, preset in enumerate(self.presets):
            hint = self._hotkey_hint(index)
            label_text = f"{hint}{preset.name}"
            if metrics.horizontalAdvance(label_text) > text_width:
                label_text = metrics.elidedText(label_text, Qt.TextElideMode.ElideRight, max(20, text_width))
            card = QPushButton(label_text)
            id_chip = QLabel(f"#{preset.class_id}", card)
            id_chip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            id_chip.setStyleSheet("color: #8B9099; font-size: 11px; font-weight: 400; border: none; background: transparent;")
            chip_row = QHBoxLayout(card); chip_row.setContentsMargins(0, 0, 9, 0); chip_row.setSpacing(0)
            chip_row.addStretch(1); chip_row.addWidget(id_chip); card.setObjectName("labelCard"); card.setFixedHeight(36); card.setMinimumWidth(0); card.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed); card.setCursor(Qt.CursorShape.PointingHandCursor); card.setProperty("selected", False); card.setProperty("preset_index", index)
            card.setStyleSheet("QPushButton { background: #34373C; color: #E6E9ED; border: 1px solid #45494F; border-left: 3px solid #45494F; border-radius: 5px; padding: 2px 6px; font-weight: 600; text-align: left; padding-left: 9px; padding-right: 26px; } QPushButton:hover { background: #3C4046; border-left: 3px solid #6A84B8; } QPushButton[selected=\"true\"] { background: #31436B; border: 1px solid #6A84B8; border-left: 3px solid #7FA3E0; color: #FFFFFF; }")
            card.clicked.connect(lambda checked=False, row=index: self.list.setCurrentRow(row)); card.installEventFilter(self)
            row, column = divmod(index, columns); self.grid.addWidget(card, row, column)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.presets and getattr(self, "_last_seen_columns", None) != self._preferred_columns():
            self._last_seen_columns = self._preferred_columns()
            self._render_presets()

    def eventFilter(self, watched, event) -> bool:
        kind = event.type()
        if watched.property("preset_index") is not None and kind == QEvent.Type.MouseButtonDblClick: self.edit_preset(int(watched.property("preset_index"))); return True
        if watched in {self.label_frame, self.grid_host, self.label_scroll.viewport()} and kind == QEvent.Type.MouseButtonPress: self.clear_selection()
        return super().eventFilter(watched, event)

    def _selected(self, row: int) -> None:
        if 0 <= row < len(self.presets): self.select_label(self.presets[row].name, emit=False); self.presetSelected.emit(self.presets[row].name)

    def select_label(self, label: str, emit: bool = False) -> None:
        self._selected_name = label
        for index, preset in enumerate(self.presets):
            item = self.grid.itemAtPosition(index // max(1, getattr(self, "grid_columns", 3)), index % max(1, getattr(self, "grid_columns", 3)))
            if not item or not item.widget(): continue
            selected = preset.name == label; item.widget().setProperty("selected", selected); item.widget().style().unpolish(item.widget()); item.widget().style().polish(item.widget())
            if selected:
                self.list.blockSignals(True); self.list.setCurrentRow(index); self.list.blockSignals(False)
                if emit: self.presetSelected.emit(preset.name)

    def clear_selection(self) -> None:
        self._selected_name = None
        self.list.blockSignals(True); self.list.clearSelection(); self.list.setCurrentRow(-1); self.list.blockSignals(False)
        for index in range(self.grid.count()):
            widget = self.grid.itemAt(index).widget()
            if widget: widget.setProperty("selected", False); widget.style().unpolish(widget); widget.style().polish(widget)

    @property
    def selected_label(self) -> str | None:
        return self._selected_name

    def label_index(self, label: str) -> int:
        for index, preset in enumerate(self.presets):
            if preset.name == label:
                return index
        return -1

    def hotkey_target(self, number: int) -> str | None:
        """What the number key currently selects: bound label or default slot."""
        if number in self._hotkey_bindings:
            return self._hotkey_bindings[number]
        if 1 <= number <= len(self.presets):
            return self.presets[number - 1].name
        return None

    def label_bound_number(self, label: str) -> int | None:
        for number, bound in self._hotkey_bindings.items():
            if bound == label:
                return number
        return None

    def bind_hotkey(self, number: int, label: str) -> None:
        """Bind one number key to one label; a number only ever maps to one."""
        if not label:
            return
        self._hotkey_bindings[number] = label
        QSettings("RelinRan", "ModelLabeling").setValue(f"hotkey/{number}", label)
        self._render_presets()

    def _clear_hotkey_for(self, label: str) -> None:
        for number in list(self._hotkey_bindings):
            if self._hotkey_bindings[number] == label:
                self._hotkey_bindings.pop(number)
                QSettings("RelinRan", "ModelLabeling").remove(f"hotkey/{number}")

    def _hotkey_hint(self, index: int) -> str:
        """Key-number shown on a card: bound number first, then the default
        slot when that number is still unbound."""
        preset = self.presets[index]
        for number, label in self._hotkey_bindings.items():
            if label == preset.name:
                return f"{number}· "
        if index < 9 and (index + 1) not in self._hotkey_bindings:
            return f"{index + 1}· "
        return ""

    def _edit_dialog(self, row: int) -> LabelEditDialog:
        preset = self.presets[row] if 0 <= row < len(self.presets) else None; group = self.selected_group()
        return LabelEditDialog(group.name if group else ("Labels" if self.language == "en_US" else "\u6807\u7b7e\u7ec4"), preset.name if preset else f"label_{max((p.class_id for p in self.presets), default=-1) + 1}", preset.class_id if preset else max((p.class_id for p in self.presets), default=-1) + 1, self, self.language)

    def add_preset(self) -> None:
        dialog = self._edit_dialog(-1)
        if dialog.exec() != dialog.DialogCode.Accepted: return
        name = dialog.name_edit.text().strip()
        if not name or any(item.name == name for item in self.presets): return
        self.presets.append(LabelPreset(name, dialog.class_id.value(), "#00e5ff")); self._render_presets(); self.select_label(name, emit=True); self.presetsChanged.emit(); self.groupsChanged.emit()

    def edit_preset(self, row: int) -> None:
        if not 0 <= row < len(self.presets): return
        preset = self.presets[row]
        dialog = self._edit_dialog(row)
        if dialog.exec() != dialog.DialogCode.Accepted: return
        name = dialog.name_edit.text().strip()
        if not name or any(item.name == name and index != row for index, item in enumerate(self.presets)): return
        self.presets[row].name = name; self.presets[row].class_id = dialog.class_id.value(); self._render_presets(); self.presetsChanged.emit(); self.groupsChanged.emit()

    def edit_selected(self) -> None:
        row = self.list.currentRow()
        # Keep the original non-visible test workflow non-modal.
        if not self.isVisible() and 0 <= row < len(self.presets):
            preset = self.presets[row]
            name, accepted = QInputDialog.getText(self, "Edit Label", "Name", text=preset.name)
            if accepted and name.strip():
                class_id, accepted = QInputDialog.getInt(self, "Edit Label", "Class ID", preset.class_id, 0, 9999)
                if accepted:
                    preset.name = name.strip(); preset.class_id = class_id; self._render_presets(); self.select_label(preset.name); self.presetsChanged.emit(); self.groupsChanged.emit()
            return
        self.edit_preset(row)

    def delete_selected(self) -> None:
        row = self.list.currentRow()
        if not 0 <= row < len(self.presets): return
        preset = self.presets[row]
        if preset.name in self._protected_labels:
            english = self.language == "en_US"
            AppDialog.information(
                "提示",
                f"标签“{preset.name}”已被点位类型使用，无法删除。" if not english
                else f"Label '{preset.name}' is used by a keypoint type and cannot be deleted.",
                self,
            )
            return
        if len(self.presets) <= 1:
            english = self.language == "en_US"
            AppDialog.information(
                "提示",
                "每个标签分组至少需要保留一个标签，可编辑但不能删除。" if not english
                else "A label group must keep at least one label; edit it instead of deleting.",
                self,
            )
            return
        if not AppDialog.question("提示" if self.language == "en_US" else "\u5220\u9664\u6807\u7b7e", f"Delete label '{preset.name}'?" if self.language == "en_US" else f"\u786e\u8ba4\u5220\u9664\u6807\u7b7e\u201c{preset.name}\u201d\u5417\uff1f", self): return
        self._clear_hotkey_for(preset.name)
        self.presets.pop(row); self._render_presets(); self.presetsChanged.emit(); self.groupsChanged.emit()
        # Re-select an adjacent row so the next delete/edit acts on a real item.
        target = min(row, len(self.presets) - 1)
        if target >= 0:
            self.select_label(self.presets[target].name)
