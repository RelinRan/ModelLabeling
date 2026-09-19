import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QListWidget

from src.models.annotation import Annotation, LabelPreset, ShapeType
from src.models.project import ImageRecord, LabelGroup
from src.widgets.label_groups_dialog import LabelGroupsDialog
from src.widgets.main_window import MainWindow
from src.widgets.preset_panel import PresetPanel


def _box(label: str) -> Annotation:
    return Annotation(ShapeType.RECTANGLE, label, [QPointF(1, 2), QPointF(20, 30)])


def test_status_bar_shows_current_file_annotation_count_after_resolution(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.settings.language = "zh_CN"
    record = ImageRecord(tmp_path / "generated.jpg", 320, 240, "JPEG", 128, [_box("one"), _box("two")])
    window.state.images = [record]
    window.state.current_index = 0
    window.dataset_current_index = 0
    window.dataset_total_images = 1

    window.refresh_stats()

    assert window.status_current_count.text() == "\u6807\u7b7e\u6570\u91cf\uff1a2"
    layout = window.status_bar.layout()
    resolution_index = next(index for index in range(layout.count()) if layout.itemAt(index).widget() is window.status_resolution)
    count_index = next(index for index in range(layout.count()) if layout.itemAt(index).widget() is window.status_current_count)
    assert count_index == resolution_index + 1

    record.annotations.pop()
    window.refresh_stats()
    assert window.status_current_count.text() == "\u6807\u7b7e\u6570\u91cf\uff1a1"
    window.close()


def test_label_group_and_preset_items_cannot_be_dragged_or_reordered():
    app = QApplication.instance() or QApplication([])
    groups = [
        LabelGroup("first", [LabelPreset("one", 0, "#ffffff")]),
        LabelGroup("second", [LabelPreset("two", 1, "#000000")]),
    ]
    dialog = LabelGroupsDialog(groups)
    panel = PresetPanel()

    for widget in (dialog.group_list, panel.list):
        assert widget.dragDropMode() == QListWidget.DragDropMode.NoDragDrop
        assert not widget.dragEnabled()
        assert not widget.acceptDrops()
        assert not widget.showDropIndicator()

    assert [dialog.group_list.item(index).text() for index in range(dialog.group_list.count())] == ["first", "second"]
    dialog.close()
    panel.close()
