import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.models.annotation import ShapeType
from src.models.project import ProjectSettings
from src.widgets.settings_dialog import SettingsDialog


def _enabled_shapes(dialog: SettingsDialog) -> set[ShapeType]:
    return {
        ShapeType(dialog.shape_combo.itemData(index))
        for index in range(dialog.shape_combo.count())
        if dialog.shape_combo.model().item(index).isEnabled()
    }


def test_settings_shape_options_follow_official_dataset_task_capabilities():
    app = QApplication.instance() or QApplication([])
    dialog = SettingsDialog(ProjectSettings(language="en_US"))

    cases = {
        "yolo_detection": {ShapeType.RECTANGLE, ShapeType.SQUARE},
        "yolo_segmentation": {ShapeType.RECTANGLE, ShapeType.SQUARE, ShapeType.POLYGON},
        "yolo_pose": {ShapeType.KEYPOINT},
        "yolo_obb": {ShapeType.OBB},
    }
    for task, expected in cases.items():
        dialog.task.setCurrentIndex(dialog.task.findData(task))
        assert _enabled_shapes(dialog) == expected

    dialog.format.setCurrentIndex(dialog.format.findData("coco"))
    # Standard COCO has no rotated box; OBB is YOLO-only.
    assert _enabled_shapes(dialog) == set(ShapeType) - {ShapeType.OBB}
    dialog.close()


def test_keypoint_shape_selection_is_persisted():
    app = QApplication.instance() or QApplication([])
    settings = ProjectSettings(
        annotation_format="yolo",
        dataset_task="yolo_pose",
        enabled_shapes=[ShapeType.KEYPOINT],
        language="en_US",
    )
    dialog = SettingsDialog(settings)

    assert dialog.shape_combo.currentData() == ShapeType.KEYPOINT
    applied = dialog.apply()

    assert applied.dataset_task == "yolo_pose"
    assert applied.enabled_shapes == [ShapeType.KEYPOINT]
    dialog.close()
