import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.models.annotation import Annotation, ShapeType
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
    assert not view.draw_enabled
    view.close()
