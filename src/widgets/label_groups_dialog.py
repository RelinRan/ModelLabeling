from __future__ import annotations

from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QListWidget, QPushButton, QVBoxLayout
from src.models.annotation import LabelPreset
from src.models.project import LabelGroup
from .common_dialogs import AppDialog, NameDialog
from .preset_panel import PresetPanel


class LabelGroupsDialog(QDialog):
    def __init__(self, groups: list[LabelGroup], parent=None, language: str = "zh_CN") -> None:
        super().__init__(parent)
        self.language = language; english = language == "en_US"
        self.setWindowTitle("Label Groups" if english else "\u6807\u7b7e\u5206\u7ec4"); self.setMinimumSize(760, 500)
        self.groups = [
            LabelGroup(
                group.name,
                [LabelPreset.from_dict(preset.to_dict()) for preset in group.presets],
                group.protected,
            )
            for group in groups
        ]
        self.group_list = QListWidget(); self.group_list.currentRowChanged.connect(self._select_group)
        self.panel = PresetPanel(language=language); self.panel.groupsChanged.connect(self._sync_group)
        self.add_group_button = QPushButton("Add Group" if english else "\u65b0\u589e\u5206\u7ec4")
        self.edit_group_button = QPushButton("Edit Group" if english else "\u7f16\u8f91\u5206\u7ec4")
        self.delete_group_button = QPushButton("Delete Group" if english else "\u5220\u9664\u5206\u7ec4")
        for button, handler in ((self.add_group_button, self.add_group), (self.edit_group_button, self.edit_group), (self.delete_group_button, self.delete_group)): button.clicked.connect(handler)
        left = QFrame(); left.setObjectName("labelArea")
        left_layout = QVBoxLayout(left); left_layout.setContentsMargins(8, 8, 8, 8); left_layout.setSpacing(8); left_layout.addWidget(self.group_list)
        group_actions = QHBoxLayout(); group_actions.addWidget(self.add_group_button); group_actions.addWidget(self.edit_group_button); group_actions.addWidget(self.delete_group_button); left_layout.addLayout(group_actions)
        right = QFrame(); right.setObjectName("labelGroupsRight")
        right.setStyleSheet(
            "QFrame#labelGroupsRight { background: #25272A; border: 1px solid #464A50; border-radius: 5px; }"
        )
        right_layout = QVBoxLayout(right); right_layout.setContentsMargins(8, 8, 8, 8); right_layout.setSpacing(8); right_layout.addWidget(self.panel)
        # The left group list is the selector in this dialog, so the panel's
        # duplicate group combo is unnecessary. Keep the confirmation action
        # with the label actions.
        self.panel.group_combo.hide()
        self.panel.layout().removeWidget(self.panel.group_combo)
        panel_layout = self.panel.layout()
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(8)
        self.confirm_button = QPushButton("Confirm" if english else "\u786e\u8ba4")
        self.confirm_button.clicked.connect(self._confirm)
        self.panel.label_frame.setStyleSheet(
            "QFrame#labelArea { background: #25272A; border: 1px solid #464A50; border-radius: 5px; }"
        )
        self.panel.button_frame.setStyleSheet("QFrame#labelActions { background: transparent; border: none; }")
        self.panel.button_frame.layout().setContentsMargins(0, 0, 0, 0)
        self.panel.button_frame.layout().setSpacing(6)
        self.panel.button_frame.layout().addWidget(self.confirm_button)
        body = QHBoxLayout(); body.setSpacing(14); body.addWidget(left, 1); body.addWidget(right, 2)
        layout = QVBoxLayout(self); layout.setContentsMargins(18, 18, 18, 18); layout.addLayout(body)
        self._refresh_groups(); self.group_list.setCurrentRow(0)

    @property
    def english(self) -> bool: return self.language == "en_US"
    def _confirm(self) -> None: self._sync_group(); self.accept()
    def _refresh_groups(self) -> None: self.group_list.clear(); self.group_list.addItems([group.name for group in self.groups])
    def _select_group(self, row: int) -> None:
        if 0 <= row < len(self.groups): self.panel.set_groups(self.groups, self.groups[row].name)
    def _sync_group(self) -> None:
        group = self.panel.selected_group()
        if group is not None: group.presets = list(self.panel.presets)

    def add_group(self) -> None:
        dialog = NameDialog("Add Label Group" if self.english else "\u65b0\u589e\u6807\u7b7e\u7ec4", parent=self, language=self.language)
        if dialog.exec() == dialog.DialogCode.Accepted:
            name = dialog.name_edit.text().strip()
            if name and all(group.name != name for group in self.groups): self.groups.append(LabelGroup(name, [])); self._refresh_groups(); self.group_list.setCurrentRow(len(self.groups) - 1)

    def edit_group(self) -> None:
        row = self.group_list.currentRow()
        if row < 0: return
        dialog = NameDialog("Edit Label Group" if self.english else "\u7f16\u8f91\u6807\u7b7e\u7ec4", self.groups[row].name, self, self.language)
        if dialog.exec() == dialog.DialogCode.Accepted:
            name = dialog.name_edit.text().strip()
            if name: self.groups[row].name = name; self._refresh_groups(); self.group_list.setCurrentRow(row)

    def delete_group(self) -> None:
        row = self.group_list.currentRow()
        if row < 0: return
        if self.groups[row].protected: AppDialog.information("Label Groups" if self.english else "\u6807\u7b7e\u5206\u7ec4", "The default label group cannot be deleted." if self.english else "\u9ed8\u8ba4\u6807\u7b7e\u7ec4\u4e0d\u53ef\u5220\u9664\u3002", self); return
        name = self.groups[row].name
        if not AppDialog.question("Delete Label Group" if self.english else "\u5220\u9664\u6807\u7b7e\u7ec4", f"Delete label group '{name}'?" if self.english else f"\u786e\u8ba4\u5220\u9664\u6807\u7b7e\u7ec4\u201c{name}\u201d\u5417\uff1f", self): return
        self.groups.pop(row); self._refresh_groups(); self.group_list.setCurrentRow(max(0, row - 1))
