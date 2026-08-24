from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout

from src.models.annotation import Annotation, Keypoint
from src.models.project import LabelGroup
from .form_layout import BUTTON_TOP_SPACING, configure_buttons, configure_form, set_confirm_button, set_content_margins, size_buttons


class AnnotationEditDialog(QDialog):
    """Edit an annotation label and, for pose objects, its keypoint values."""

    def __init__(self, groups: list[LabelGroup], current_label: str, parent=None, language: str = "zh_CN", annotation: Annotation | None = None) -> None:
        super().__init__(parent)
        self.english = language == "en_US"
        self.setWindowTitle("Edit Annotation" if self.english else "编辑标注")
        self.setFixedWidth(300)
        self.groups = groups
        self.annotation = annotation
        self.deleted = False
        self.group_combo = QComboBox()
        self.group_combo.addItems([group.name for group in groups])
        self.label_combo = QComboBox()
        self.group_combo.currentIndexChanged.connect(self._load_labels)
        for index, group in enumerate(groups):
            if any(preset.name == current_label for preset in group.presets):
                self.group_combo.setCurrentIndex(index)
                break
        self._load_labels(self.group_combo.currentIndex())
        self.label_combo.setCurrentText(current_label)
        form = configure_form(QFormLayout())
        form.addRow("Label Group" if self.english else "标签组", self.group_combo)
        form.addRow("Label" if self.english else "标签", self.label_combo)
        self.keypoint_table: QTableWidget | None = None
        self.validation_label = QLabel()
        self.validation_label.setStyleSheet("color: #ff6b6b;")
        self.validation_label.setWordWrap(True)
        if annotation is not None and annotation.keypoints:
            self.keypoint_table = self._build_keypoint_table(annotation.keypoints)
            form.addRow("Keypoints" if self.english else "关键点", self.keypoint_table)
            form.addRow("", self.validation_label)
        self.confirm_button = QPushButton("Confirm" if self.english else "确认")
        self.cancel_button = QPushButton("Cancel" if self.english else "取消")
        self.delete_button = QPushButton("Delete" if self.english else "删除")
        self.confirm_button.clicked.connect(self._accept)
        self.cancel_button.clicked.connect(self.reject)
        self.delete_button.clicked.connect(self._delete)
        buttons = configure_buttons(QHBoxLayout())
        buttons.addStretch()
        buttons.addWidget(self.delete_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.confirm_button)
        size_buttons(self.delete_button, self.cancel_button, self.confirm_button)
        set_confirm_button(self.confirm_button)
        layout = QVBoxLayout(self)
        set_content_margins(layout)
        layout.setSpacing(0)
        layout.addLayout(form)
        layout.addSpacing(BUTTON_TOP_SPACING)
        layout.addLayout(buttons)
        self.adjustSize()

    def _load_labels(self, index: int) -> None:
        self.label_combo.clear()
        if 0 <= index < len(self.groups):
            for preset in self.groups[index].presets:
                self.label_combo.addItem(preset.name)

    def _build_keypoint_table(self, keypoints: list[Keypoint]) -> QTableWidget:
        table = QTableWidget(len(keypoints), 4, self)
        table.setHorizontalHeaderLabels(["Name" if self.english else "名称", "X", "Y", "Visibility" if self.english else "可见性"])
        table.verticalHeader().setVisible(False)
        table.setMinimumHeight(min(190, 28 + len(keypoints) * 24))
        table.setMaximumHeight(230)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setStretchLastSection(True)
        for row, keypoint in enumerate(keypoints):
            name = QTableWidgetItem(keypoint.name)
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, name)
            for column, value in ((1, keypoint.point.x()), (2, keypoint.point.y())):
                field = QLineEdit(f"{value:.2f}", table)
                field.setValidator(QDoubleValidator(-100000000.0, 100000000.0, 4, field))
                field.setAlignment(Qt.AlignmentFlag.AlignRight)
                table.setCellWidget(row, column, field)
            visibility = QComboBox(table)
            visibility.addItem("0 - " + ("absent" if self.english else "未标注"), 0)
            visibility.addItem("1 - " + ("occluded" if self.english else "遮挡"), 1)
            visibility.addItem("2 - " + ("visible" if self.english else "可见"), 2)
            visibility.setCurrentIndex(keypoint.visibility)
            table.setCellWidget(row, 3, visibility)
        table.setColumnWidth(0, 76)
        table.setColumnWidth(1, 54)
        table.setColumnWidth(2, 54)
        return table

    def _accept(self) -> None:
        if self.keypoint_table is not None and self.annotation is not None:
            edited: list[Keypoint] = []
            for row, original in enumerate(self.annotation.keypoints):
                x_field = self.keypoint_table.cellWidget(row, 1)
                y_field = self.keypoint_table.cellWidget(row, 2)
                visibility = self.keypoint_table.cellWidget(row, 3)
                try:
                    x = float(x_field.text())
                    y = float(y_field.text())
                except (TypeError, ValueError):
                    self.validation_label.setText("Enter valid keypoint coordinates" if self.english else "请输入有效的关键点坐标")
                    return
                edited.append(Keypoint(original.name, QPointF(x, y), int(visibility.currentData())))
            self.annotation.keypoints = edited
        self.accept()

    def _delete(self) -> None:
        self.deleted = True
        self.accept()

    def selected_label(self) -> str:
        return self.label_combo.currentText()
