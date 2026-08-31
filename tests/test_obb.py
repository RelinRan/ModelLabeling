import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import math
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.models.annotation import Annotation, ShapeType
from src.models.project import ProjectSettings
from src.services.annotation_service import AnnotationService
from src.services.dataset_detector import DatasetDetector
from src.services.format_capabilities import CAPABILITIES, DatasetTask, UnsupportedAnnotationError, validate_annotations
from src.widgets.canvas_view import CanvasView
from PIL import Image


def _make_image(path: Path, width=320, height=240) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (30, 30, 30)).save(path)


def test_obb_task_capabilities_and_validation():
    capabilities = CAPABILITIES[DatasetTask.YOLO_OBB]
    assert set(capabilities.shapes) == {ShapeType.OBB}
    corners = [QPointF(10, 10), QPointF(60, 12), QPointF(58, 50), QPointF(8, 48)]
    obb = Annotation(ShapeType.OBB, "ship", corners)
    rect = Annotation(ShapeType.RECTANGLE, "ship", [QPointF(0, 0), QPointF(10, 10)])
    settings = ProjectSettings(annotation_format="yolo", dataset_task="yolo_obb")
    assert validate_annotations([obb], "yolo", "yolo_obb") is capabilities
    try:
        validate_annotations([rect], "yolo", "yolo_obb")
        raise AssertionError("rectangle must be rejected in yolo_obb")
    except UnsupportedAnnotationError:
        pass
    try:
        validate_annotations([Annotation(ShapeType.OBB, "ship", corners[:3])], "yolo", "yolo_obb")
        raise AssertionError("three corners must be rejected")
    except UnsupportedAnnotationError:
        pass
    # COCO has no rotated box in the standard.
    try:
        validate_annotations([obb], "coco", None)
        raise AssertionError("OBB must be rejected in COCO")
    except UnsupportedAnnotationError:
        pass


def test_detector_recognizes_yolo_obb(tmp_path):
    # data.yaml task wins.
    root = tmp_path / "obb-yaml"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "data.yaml").write_text("task: obb\nnames: [ship]\n", encoding="utf-8")
    _make_image(root / "images" / "a.jpg")
    (root / "labels" / "a.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n", encoding="utf-8")
    assert DatasetDetector.detect(root).task_name == "yolo_obb"

    # All-9-column rows without yaml.
    root = tmp_path / "obb-rows"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    _make_image(root / "images" / "a.jpg")
    (root / "labels" / "a.txt").write_text(
        "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n0 0.2 0.2 0.4 0.2 0.4 0.4 0.2 0.4\n", encoding="utf-8"
    )
    assert DatasetDetector.detect(root).task_name == "yolo_obb"

    # Mixed row lengths stay segmentation (quadrilaterals plus triangles).
    root = tmp_path / "seg-mixed"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    _make_image(root / "images" / "a.jpg")
    (root / "labels" / "a.txt").write_text(
        "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n0 0.1 0.1 0.5 0.1 0.3 0.3\n", encoding="utf-8"
    )
    assert DatasetDetector.detect(root).task_name == "yolo_segmentation"


def test_obb_save_load_round_trip(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    _make_image(image_dir / "a.jpg", 200, 100)
    label_dir.mkdir()
    service = AnnotationService()
    settings = ProjectSettings(
        annotation_format="yolo", dataset_task="yolo_obb",
        label_presets=[],
    )
    from src.models.annotation import LabelPreset
    from PySide6.QtGui import QColor
    settings.label_presets = [LabelPreset("ship", 0, "#00e5ff")]
    corners = [QPointF(20, 10), QPointF(120, 14), QPointF(116, 66), QPointF(16, 62)]
    obb = Annotation(ShapeType.OBB, "ship", corners)

    service.save(image_dir / "a.jpg", [obb], label_dir, settings)
    row = (label_dir / "a.txt").read_text().splitlines()[0]
    parts = row.split()
    assert len(parts) == 9
    assert parts[0] == "0"
    assert [float(v) for v in parts[1:3]] == [0.1, 0.1]

    loaded = service.load(image_dir / "a.jpg", label_dir, settings)
    assert not loaded.error, loaded.error
    assert len(loaded.annotations) == 1
    result = loaded.annotations[0]
    assert result.shape_type == ShapeType.OBB
    for got, want in zip(result.points, corners):
        assert abs(got.x() - want.x()) < 0.01 and abs(got.y() - want.y()) < 0.01


def _corners_form_rectangle(points) -> bool:
    """True when the four points still form a rigid rotated rectangle."""
    sides = []
    for index in range(4):
        a, b = points[index], points[(index + 1) % 4]
        sides.append(math.hypot(b.x() - a.x(), b.y() - a.y()))
    if any(side < 1e-6 for side in sides):
        return False
    if abs(sides[0] - sides[2]) > 0.01 or abs(sides[1] - sides[3]) > 0.01:
        return False
    for index in range(4):
        a, b, c = points[index], points[(index + 1) % 4], points[(index + 2) % 4]
        v1 = (b.x() - a.x(), b.y() - a.y())
        v2 = (c.x() - b.x(), c.y() - b.y())
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        if abs(dot) > 0.01:
            return False
    return True


def test_obb_draw_rotate_and_undo():
    app = QApplication.instance() or QApplication([])
    view = CanvasView()
    view.resize(800, 600)
    view.show()
    view.load_image(QImage(640, 480, QImage.Format.Format_RGB32), [])
    view.set_enabled_shapes({ShapeType.OBB})
    view.set_mode(ShapeType.OBB)
    view._enable_draw_mode()
    app.processEvents()

    def point(x, y):
        return view.mapFromScene(QPointF(x, y))

    # Drag draws a rotated box starting axis-aligned (four corners).
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(100, 100))
    QTest.mouseMove(view.viewport(), point(260, 200))
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(260, 200))
    assert len(view.annotations) == 1
    obb = view.annotations[0]
    assert obb.shape_type == ShapeType.OBB
    assert len(obb.points) == 4
    assert _corners_form_rectangle(obb.points)
    assert view.draw_enabled

    # Select it, then drag the rotation handle to rotate around the center.
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, point(150, 150))
    assert view.annotation_items[0].isSelected()
    assert view.annotation_items[0].rotation_handle.isVisible()
    handle_pos = view.annotation_items[0].rotation_handle.pos()
    center = QPointF(sum(p.x() for p in obb.points) / 4, sum(p.y() for p in obb.points) / 4)
    start_angle = math.atan2(handle_pos.y() - center.y(), handle_pos.x() - center.x())
    target_angle = start_angle + math.pi / 6  # rotate 30 degrees
    target = QPointF(
        center.x() + math.cos(target_angle) * 40,
        center.y() + math.sin(target_angle) * 40,
    )
    start_view = view.mapFromScene(handle_pos)
    target_view = view.mapFromScene(target)
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start_view)
    QTest.mouseMove(view.viewport(), target_view)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, target_view)
    assert _corners_form_rectangle(obb.points)
    new_handle_pos = view.annotation_items[0].rotation_handle.pos()
    assert abs(new_handle_pos.x() - handle_pos.x()) > 0.5 or abs(new_handle_pos.y() - handle_pos.y()) > 0.5
    # Undo restores the axis-aligned corners.
    before_corners = [(p.x(), p.y()) for p in obb.points]
    QTest.keyClick(view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    restored = view.annotations[0]
    assert [(p.x(), p.y()) for p in restored.points] != before_corners
    assert _corners_form_rectangle(restored.points)
    view.close()
