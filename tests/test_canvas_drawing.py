import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.models.annotation import Annotation, Keypoint, ShapeType
from src.widgets.canvas_view import CanvasView


def test_drawing_is_disabled_until_w_and_polygon_is_click_to_finish():
    app = QApplication.instance() or QApplication([])
    view = CanvasView()
    view.resize(800, 600)
    view.show()
    view.load_image(QImage(640, 480, QImage.Format.Format_RGB32), [])
    app.processEvents()

    def point(x, y):
        return view.mapFromScene(QPointF(x, y))

    # Ordinary drag is navigation, not annotation creation.
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(20, 20))
    QTest.mouseMove(view.viewport(), point(100, 100))
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(100, 100))
    assert not view.annotations

    view.set_mode(ShapeType.POLYGON)
    QTest.keyClick(view, Qt.Key.Key_W)
    assert view.draw_enabled
    for x, y in ((50, 50), (180, 50), (100, 180)):
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(x, y))
    QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(100, 180))
    assert len(view.annotations) == 1
    assert view.annotations[0].shape_type == ShapeType.POLYGON
    assert len(view.annotations[0].points) == 3
    # Conventional behavior: the selected method stays armed for the next
    # shape until the user exits with Esc / W / right click.
    assert view.draw_enabled
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert not view.draw_enabled

    view.close()


def test_multipart_polygon_renders_and_secondary_part_is_selectable():
    app = QApplication.instance() or QApplication([])
    parts = [
        [QPointF(20, 20), QPointF(80, 20), QPointF(50, 80)],
        [QPointF(200, 200), QPointF(260, 200), QPointF(230, 260)],
    ]
    annotation = Annotation(
        ShapeType.POLYGON, "person", parts[0], polygon_parts=parts,
    )
    view = CanvasView()
    view.resize(800, 600)
    view.show()
    view.load_image(QImage(640, 480, QImage.Format.Format_RGB32), [annotation])
    app.processEvents()

    item = view.annotation_items[0]
    assert len(item.polygon_part_items) == 1
    secondary_center = view.mapFromScene(QPointF(230, 220))
    hit = view.itemAt(secondary_center)
    while hit and hit is not item:
        hit = hit.parentItem()
    assert hit is item

    view.close()


def test_custom_keypoint_schema_finishes_after_configured_point_count():
    app = QApplication.instance() or QApplication([])
    view = CanvasView()
    view.resize(800, 600)
    view.show()
    view.load_image(QImage(640, 480, QImage.Format.Format_RGB32), [])
    view.set_enabled_shapes({ShapeType.KEYPOINT})
    view.set_mode(ShapeType.KEYPOINT)
    view.set_keypoint_schema(["start", "end"])
    app.processEvents()

    view._enable_draw_mode()
    for point in (QPointF(100, 100), QPointF(200, 200)):
        QTest.mouseClick(
            view.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, view.mapFromScene(point),
        )

    assert len(view.annotations) == 1
    assert [item.name for item in view.annotations[0].keypoints] == ["start", "end"]
    assert view.draw_enabled
    view.close()


def test_method_combo_lists_only_enabled_methods_and_arms_drawing():
    app = QApplication.instance() or QApplication([])
    view = CanvasView()
    view.resize(800, 600)
    view.show()
    view.load_image(QImage(640, 480, QImage.Format.Format_RGB32), [])
    view.set_enabled_shapes({ShapeType.RECTANGLE, ShapeType.SQUARE})
    app.processEvents()

    shapes = [view.method_combo.itemData(i) for i in range(view.method_combo.count())]
    assert shapes == [ShapeType.RECTANGLE, ShapeType.SQUARE]
    assert not view.method_combo.isVisibleTo(view) or not view.draw_enabled

    requested = []
    view.methodRequested.connect(lambda shape: requested.append(shape))
    view.method_combo.setCurrentIndex(1)  # pick Square as a user would
    assert view.mode == ShapeType.SQUARE
    assert view.draw_enabled
    assert requested == [ShapeType.SQUARE]

    # Unsupported methods never appear even after a language switch.
    view.set_language("en_US")
    labels = [view.method_combo.itemText(i) for i in range(view.method_combo.count())]
    assert labels == ["Rectangle", "Square"]

    view.close()


def test_polygon_enter_finishes_and_backspace_removes_last_vertex():
    app = QApplication.instance() or QApplication([])
    view = CanvasView()
    view.resize(800, 600)
    view.show()
    view.load_image(QImage(640, 480, QImage.Format.Format_RGB32), [])
    view.set_enabled_shapes({ShapeType.POLYGON})
    view.set_mode(ShapeType.POLYGON)
    view._enable_draw_mode()
    app.processEvents()

    def point(x, y):
        return view.mapFromScene(QPointF(x, y))

    for x, y in ((50, 50), (180, 50), (100, 180), (300, 300)):
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(x, y))
    QTest.keyClick(view, Qt.Key.Key_Backspace)
    assert len(view.polygon_points) == 3
    QTest.keyClick(view, Qt.Key.Key_Return)
    assert len(view.annotations) == 1
    assert len(view.annotations[0].points) == 3
    assert view.draw_enabled
    view.close()


def test_undo_redo_restores_draw_delete_and_label_changes():
    app = QApplication.instance() or QApplication([])
    view = CanvasView()
    view.resize(800, 600)
    view.show()
    view.load_image(QImage(640, 480, QImage.Format.Format_RGB32), [])
    view.set_enabled_shapes({ShapeType.RECTANGLE})
    view.set_mode(ShapeType.RECTANGLE)
    view._enable_draw_mode()
    app.processEvents()

    def point(x, y):
        return view.mapFromScene(QPointF(x, y))

    def draw_box(x1, y1, x2, y2):
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(x1, y1))
        QTest.mouseMove(view.viewport(), point(x2, y2))
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(x2, y2))

    draw_box(20, 20, 200, 120)
    draw_box(300, 100, 500, 300)
    assert len(view.annotations) == 2

    QTest.keyClick(view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert len(view.annotations) == 1
    QTest.keyClick(view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert len(view.annotations) == 0
    assert not view.undo()

    QTest.keyClick(view, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    assert len(view.annotations) == 1
    QTest.keyClick(view, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    assert len(view.annotations) == 2
    assert not view.redo()

    # Deleting is undoable, and undo restores the same geometry.
    view.scene.clearSelection()
    view.annotation_items[0].setSelected(True)
    rect_before = QRectF(view.annotations[0].points[0], view.annotations[0].points[-1])
    QTest.keyClick(view, Qt.Key.Key_Delete)
    assert len(view.annotations) == 1
    QTest.keyClick(view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert len(view.annotations) == 2
    rect_after = QRectF(view.annotations[0].points[0], view.annotations[0].points[-1])
    assert rect_before.toRect() == rect_after.toRect()

    # Label change is undoable.
    view.scene.clearSelection()
    view.annotation_items[1].setSelected(True)
    assert view.update_selected_label("other")
    assert view.annotations[1].label == "other"
    view.undo()
    assert view.annotations[1].label != "other"
    view.close()


def test_shift_constrains_rectangle_to_square():
    app = QApplication.instance() or QApplication([])
    view = CanvasView()
    view.resize(800, 600)
    view.show()
    view.load_image(QImage(640, 480, QImage.Format.Format_RGB32), [])
    view.set_enabled_shapes({ShapeType.RECTANGLE})
    view.set_mode(ShapeType.RECTANGLE)
    view._enable_draw_mode()
    app.processEvents()

    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, view.mapFromScene(QPointF(50, 50)))
    QTest.mouseMove(view.viewport(), view.mapFromScene(QPointF(250, 150)))
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, view.mapFromScene(QPointF(250, 150)))

    assert len(view.annotations) == 1
    annotation = view.annotations[0]
    rect = QRectF(annotation.points[0], annotation.points[-1]).normalized()
    assert annotation.shape_type == ShapeType.RECTANGLE
    assert abs(rect.width() - rect.height()) < 0.001
    view.close()


def test_polygon_right_click_finishes_and_escape_is_two_stage():
    app = QApplication.instance() or QApplication([])
    view = CanvasView()
    view.resize(800, 600)
    view.show()
    view.load_image(QImage(640, 480, QImage.Format.Format_RGB32), [])
    view.set_enabled_shapes({ShapeType.POLYGON})
    view.set_mode(ShapeType.POLYGON)
    view._enable_draw_mode()
    app.processEvents()

    def point(x, y):
        return view.mapFromScene(QPointF(x, y))

    for x, y in ((60, 60), (200, 60), (130, 190)):
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(x, y))
    QTest.mouseClick(view.viewport(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, point(130, 190))
    assert len(view.annotations) == 1
    assert view.annotations[0].shape_type == ShapeType.POLYGON
    assert view.draw_enabled

    # First Esc while a shape is in progress drops it but stays armed.
    for x, y in ((320, 60), (460, 60), (390, 190)):
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(x, y))
    assert view.polygon_points
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert view.draw_enabled
    assert not view.polygon_points
    # Second Esc exits the method.
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert not view.draw_enabled
    view.close()


def test_keypoint_count_stepper_changes_schema_and_finish_threshold():
    app = QApplication.instance() or QApplication([])
    view = CanvasView()
    view.resize(800, 600)
    view.show()
    view.load_image(QImage(640, 480, QImage.Format.Format_RGB32), [])
    view.set_enabled_shapes({ShapeType.KEYPOINT})
    view.set_mode(ShapeType.KEYPOINT)
    app.processEvents()

    view.set_keypoint_count(3)
    assert len(view.keypoint_schema) == 3
    assert view.keypoint_count_box.value() == 3

    emitted = []
    view.keypointCountChanged.connect(lambda value: emitted.append(value))
    view.keypoint_count_box.setValue(5)
    assert len(view.keypoint_schema) == 5
    assert emitted == [5]
    # Drawing auto-finishes exactly at the configured count.
    view._enable_draw_mode()
    for point in (QPointF(60, 60), QPointF(160, 60), QPointF(110, 160), QPointF(220, 160), QPointF(300, 120)):
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, view.mapFromScene(point))
    assert len(view.annotations) == 1
    assert len(view.annotations[0].keypoints) == 5
    # 17 keeps the official COCO person names.
    view.set_keypoint_count(17)
    assert view.keypoint_schema[0] == "nose"
    view.close()


def test_write_kpt_shape_roundtrip(tmp_path):
    from src.services.yolo_metadata import write_kpt_shape, yolo_keypoint_shape, yolo_keypoint_names, load_yolo_metadata

    # Creates a fresh data.yaml with pose task.
    path = write_kpt_shape(tmp_path, 6, 3, ["hl", "hr", "fl", "fr", "nose", "tail"])
    assert path == tmp_path / "data.yaml"
    assert yolo_keypoint_shape(tmp_path) == (6, 3)
    assert yolo_keypoint_names(tmp_path) == ["hl", "hr", "fl", "fr", "nose", "tail"]
    assert load_yolo_metadata(tmp_path)["task"] == "pose"

    # Updates an existing file and preserves unrelated fields.
    (tmp_path / "data.yaml").write_text(
        "task: detect\nnames: [cat, dog]\nkpt_shape: [2, 2]\n", encoding="utf-8"
    )
    write_kpt_shape(tmp_path, 4, 3, None)
    document = load_yolo_metadata(tmp_path)
    assert document["kpt_shape"] == [4, 3]
    assert document["names"] == ["cat", "dog"]
    assert document["task"] == "detect"  # preserved, not overwritten


def test_right_click_cycles_keypoint_visibility():
    app = QApplication.instance() or QApplication([])
    keypoint_annotation = Annotation(
        ShapeType.KEYPOINT, "person",
        [QPointF(10, 10), QPointF(60, 60)],
        keypoints=[Keypoint("nose", QPointF(20, 20), 2), Keypoint("eye", QPointF(50, 50), 2)],
    )
    view = CanvasView()
    view.resize(800, 600)
    view.show()
    view.load_image(QImage(640, 480, QImage.Format.Format_RGB32), [keypoint_annotation])
    app.processEvents()

    marker_pos = view.mapFromScene(QPointF(20, 20))
    QTest.mouseClick(view.viewport(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, marker_pos)
    assert view.annotations[0].keypoints[0].visibility == 0
    QTest.mouseClick(view.viewport(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, marker_pos)
    assert view.annotations[0].keypoints[0].visibility == 1
    QTest.keyClick(view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert view.annotations[0].keypoints[0].visibility == 0
    QTest.keyClick(view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert view.annotations[0].keypoints[0].visibility == 2
    view.close()


def test_wheel_zoom_and_view_persistence_across_images():
    app = QApplication.instance() or QApplication([])
    view = CanvasView()
    view.resize(800, 600)
    view.show()
    view.load_image(QImage(640, 480, QImage.Format.Format_RGB32), [])
    app.processEvents()
    fit_scale = view.fit_scale

    # Wheel zoom keeps the cursor anchor and clamps.
    from PySide6.QtCore import QPoint, QEvent
    from PySide6.QtGui import QWheelEvent
    center = QPoint(400, 300)
    before = view.mapToScene(center)
    wheel = QWheelEvent(QPointF(center), QPointF(view.mapToGlobal(center)), QPoint(0, 0), QPoint(0, 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    view.wheelEvent(wheel)
    after = view.mapToScene(center)
    assert view.transform().m11() > fit_scale
    assert (after - before).manhattanLength() < 1.0  # anchored at the cursor

    # Zoom survives switching to another image of the same aspect.
    ratio = view.transform().m11() / view.fit_scale
    view.load_image(QImage(320, 240, QImage.Format.Format_RGB32), [])
    app.processEvents()
    assert abs(view.transform().m11() / view.fit_scale - ratio) < 0.02

    # An explicit fit resets the memory.
    view.fit_image()
    view.load_image(QImage(320, 240, QImage.Format.Format_RGB32), [])
    app.processEvents()
    assert abs(view.transform().m11() / view.fit_scale - 1.0) < 1e-6
    view.close()
