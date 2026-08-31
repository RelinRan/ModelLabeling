"""End-to-end annotation test: open 3 dataset formats, draw 4 shape types.

Drives the real MainWindow and CanvasView with synthetic Qt mouse events
(the same event path real mouse input takes), waits for the background
scan/save workers, and verifies the written annotation files.

Datasets under E:\\Dataset\\multiple\\test:
  yolo-action-test  (YOLO Detection) -> rectangle + square
  voc-action-test   (Pascal VOC)     -> rectangle + square
  coco-action-test  (COCO)           -> rectangle + polygon + keypoint

Usage: python -u test_annotation_flow.py
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from src.models.annotation import ShapeType
from src.models.keypoint import COCO_PERSON_KEYPOINTS
from src.models.project import KeypointGroup
from src.services.format_capabilities import CAPABILITIES, task_for_format
from src.widgets.canvas_view import AnnotationItem, CanvasView
from src.widgets.common_dialogs import AppDialog
from src.widgets.main_window import MainWindow

ROOT = Path(__file__).resolve().parent
SHOTS = ROOT / "test_shots"
TEST_ROOT = Path(r"E:\Dataset\multiple\test")
# A fully unannotated YOLO layout is created at runtime for the fresh-dataset
# scenario; its images are copied from an existing dataset.
FRESH_ROOT = Path(tempfile.mkdtemp(prefix="ml_fresh_")) / "fresh-yolo"
FRESH_SOURCE = TEST_ROOT / "yolo-action-test" / "images"

DATASETS = [
    ("yolo-action-test", "yolo", "yolo_detection", [("rectangle", "person"), ("square", "head")]),
    ("voc-action-test", "voc", "voc", [("rectangle", "person"), ("square", "head")]),
    ("coco-action-test", "coco", "coco", [("rectangle", "person"), ("polygon", "hand"), ("keypoint", "foot")]),
]

failures: list[str] = []
results: list[str] = []
# A modal error dialog would block the synthetic event loop forever; record
# it instead and fail the run — an unexpected dialog is itself a regression.
dialog_errors: list[str] = []
AppDialog.information = staticmethod(
    lambda title, text, parent=None, language=None: (
        dialog_errors.append(f"{title}: {text}"), print(f"DIALOG  {title}: {text}", flush=True)
    )
)


def check(ok: bool, message: str) -> None:
    line = ("PASS" if ok else "FAIL") + "  " + message
    print(line, flush=True)
    results.append(line)
    if not ok:
        failures.append(message)


def pump(predicate, timeout_ms: float, step_ms: int = 20) -> bool:
    deadline = time.perf_counter() + timeout_ms / 1000
    while time.perf_counter() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(step_ms / 1000)
    QApplication.processEvents()
    return predicate()


def shot(window, name: str) -> None:
    SHOTS.mkdir(exist_ok=True)
    window.grab().save(str(SHOTS / f"{name}.png"))


def viewport_point(canvas: CanvasView, scene_pt: QPointF) -> tuple[QPointF, QPointF]:
    view_local = canvas.mapFromScene(QPointF(scene_pt))
    if isinstance(view_local, QPoint):  # PySide6 may return either type
        view_local = QPointF(view_local)
    global_pt = QPointF(canvas.mapToGlobal(view_local.toPoint()))
    vp_local = QPointF(canvas.viewport().mapFromGlobal(global_pt.toPoint()))
    return vp_local, global_pt


def send_mouse(canvas: CanvasView, event_type: QEvent.Type, scene_pt: QPointF, buttons) -> None:
    vp_local, global_pt = viewport_point(canvas, scene_pt)
    event = QMouseEvent(
        event_type, vp_local, global_pt,
        Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(canvas.viewport(), event)
    QApplication.processEvents()


def click(canvas: CanvasView, scene_pt: QPointF) -> None:
    send_mouse(canvas, QEvent.Type.MouseButtonPress, scene_pt, Qt.MouseButton.LeftButton)
    send_mouse(canvas, QEvent.Type.MouseButtonRelease, scene_pt, Qt.MouseButton.NoButton)


def double_click(canvas: CanvasView, scene_pt: QPointF) -> None:
    send_mouse(canvas, QEvent.Type.MouseButtonPress, scene_pt, Qt.MouseButton.LeftButton)
    send_mouse(canvas, QEvent.Type.MouseButtonRelease, scene_pt, Qt.MouseButton.NoButton)
    send_mouse(canvas, QEvent.Type.MouseButtonDblClick, scene_pt, Qt.MouseButton.LeftButton)
    send_mouse(canvas, QEvent.Type.MouseButtonRelease, scene_pt, Qt.MouseButton.NoButton)


def annotation_item_at(canvas: CanvasView, pt: QPointF) -> AnnotationItem | None:
    """Mirror CanvasView._item_at_event so selection agrees with the press."""
    view_pt = canvas.mapFromScene(QPointF(pt))
    item = canvas.itemAt(view_pt if isinstance(view_pt, QPoint) else view_pt.toPoint())
    while item is not None and not isinstance(item, AnnotationItem):
        item = item.parentItem()
    return item if isinstance(item, AnnotationItem) else None


def free_point(canvas: CanvasView, pt: QPointF) -> QPointF:
    """Grid-search a start point that hits no existing annotation item."""
    image_rect = canvas.sceneRect()
    for row in range(12):
        for col in range(16):
            candidate = QPointF(
                min(pt.x() + col * 61, image_rect.right() - 260),
                min(pt.y() + row * 47, image_rect.bottom() - 300),
            )
            if annotation_item_at(canvas, candidate) is None:
                return candidate
    return pt


def draw_box(window: MainWindow, shape: ShapeType, top_left: QPointF, size: float) -> QPointF:
    """Drag-draw a rectangle/square; returns the bottom-right scene point."""
    start = free_point(window.canvas, top_left)
    end = QPointF(start.x() + size, start.y() + (size if shape == ShapeType.SQUARE else size * 0.7))
    send_mouse(window.canvas, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    for fraction in (0.25, 0.5, 0.75, 1.0):
        at = QPointF(start.x() + (end.x() - start.x()) * fraction, start.y() + (end.y() - start.y()) * fraction)
        send_mouse(window.canvas, QEvent.Type.MouseMove, at, Qt.MouseButton.LeftButton)
    send_mouse(window.canvas, QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton)
    return end


def draw_polygon(window: MainWindow, top_left: QPointF) -> None:
    base = free_point(window.canvas, top_left)
    vertices = [
        QPointF(base.x() + 220, base.y() - 60),
        QPointF(base.x() + 340, base.y() + 120),
        QPointF(base.x() + 160, base.y() + 240),
    ]
    # Every click must land clear of existing annotation items, otherwise the
    # press starts a drag of that item instead of adding a polygon vertex.
    vertices = [free_point(window.canvas, vertex) for vertex in vertices]
    for vertex in [base] + vertices:
        click(window.canvas, vertex)
    # The press preceding the double click appends the last vertex.
    double_click(window.canvas, free_point(window.canvas, QPointF(base.x() + 20, base.y() + 110)))


def draw_keypoints(window: MainWindow, top_left: QPointF) -> None:
    # Pin the keypoint type to the default 17-point schema: the user's last
    # selected type (persisted in QSettings) would otherwise change how many
    # clicks this helper needs and what names land in the saved file.
    window.canvas.set_keypoint_groups(
        [KeypointGroup("姿态", list(COCO_PERSON_KEYPOINTS), True)], "姿态"
    )
    base = free_point(window.canvas, top_left)
    offsets = ((0, 0), (90, 40), (30, 120), (150, 160))
    points = [free_point(window.canvas, QPointF(base.x() + dx, base.y() + dy)) for dx, dy in offsets]
    for point in points:
        click(window.canvas, point)
    # The double-click press appends the final keypoint and finishes.
    double_click(window.canvas, free_point(window.canvas, QPointF(base.x() + 200, base.y() + 60)))


def enable_shape(window: MainWindow, shape: ShapeType) -> None:
    window.settings.enabled_shapes = [shape]
    window._apply_annotation_capabilities()
    window.canvas._enable_draw_mode()
    assert window.canvas.draw_enabled and window.canvas.mode == shape


def save_settled(window: MainWindow) -> bool:
    """True once no save is in flight and the dirty flag is clear.

    The save runs on a plain Python thread whose reference is kept until the
    next save replaces it, so "settled" means the thread is not alive.
    """
    thread = window._save_thread
    alive = False if thread is None else (
        thread.is_alive() if hasattr(thread, "is_alive") else thread.isRunning()
    )
    return not alive and not window.dirty


def wait_saved(window: MainWindow) -> None:
    pump(lambda: save_settled(window), 15000)


def yolo_rows(annotation_dir: Path, stem: str) -> tuple[list[str], list[str]]:
    txt = next(annotation_dir.rglob(f"{stem}.txt"), None)
    rows = [line for line in txt.read_text().splitlines() if line.strip()] if txt else []
    classes = []
    classes_file = annotation_dir / "classes.txt"
    if not classes_file.exists():
        classes_file = annotation_dir.parent / "classes.txt"
    if classes_file.exists():
        classes = [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]
    return rows, classes


def voc_rows(annotation_dir: Path, stem: str) -> tuple[int, set[str]]:
    xml_path = next(annotation_dir.rglob(f"{stem}.xml"), None)
    if not xml_path:
        return 0, set()
    root = ET.fromstring(xml_path.read_text())
    names = {node.text for node in root.iter("name") if node.text}
    return len(root.findall("object")), names


def coco_rows(annotation_dir: Path, file_name: str) -> tuple[int, set[str]]:
    # For COCO the annotation_dir setting may point at the JSON file itself.
    if annotation_dir.is_file():
        annotation_dir = annotation_dir.parent
    store = annotation_dir / ".model_labeling.sqlite3"
    if not store.exists():
        return 0, set()
    connection = sqlite3.connect(str(store))
    try:
        categories = {
            int(category_id): str(json.loads(payload).get("name", ""))
            for category_id, payload in connection.execute("SELECT id,payload FROM coco_categories")
        }
        rows = connection.execute(
            "SELECT payload FROM coco_annotations WHERE image_id = "
            "(SELECT id FROM coco_images WHERE file_name = ? OR basename = ? LIMIT 1)",
            (file_name, file_name),
        ).fetchall()
        labels = {categories.get(int(json.loads(payload).get("category_id", -1)), "") for (payload,) in rows}
    finally:
        connection.close()
    return len(rows), {label for label in labels if label}


TEST_LABELS = {"person", "head", "hand", "foot"}


def _cleanup_prior_test_annotations() -> None:
    """Remove annotations left by previous test runs so each run starts clean.

    The drawn labels are not part of the datasets' original class lists, so
    they can be identified and stripped by name.
    """
    # YOLO: drop test classes from classes.txt and their rows from label files.
    yolo_root = TEST_ROOT / "yolo-action-test"
    classes = yolo_root / "classes.txt"
    if classes.exists():
        names = [line.strip() for line in classes.read_text().splitlines() if line.strip()]
        dropped = {index for index, name in enumerate(names) if name in TEST_LABELS}
        kept = [name for name in names if name not in TEST_LABELS]
        classes.write_text("\n".join(kept) + "\n", encoding="utf-8")
        for txt in (yolo_root / "labels").rglob("*.txt"):
            rows = [row for row in txt.read_text().splitlines()
                    if row.strip() and int(row.split()[0]) not in dropped]
            txt.write_text("".join(row + "\n" for row in rows), encoding="utf-8")

    # VOC: drop <object> entries whose name is a test label.
    for xml in (TEST_ROOT / "voc-action-test" / "Annotations").rglob("*.xml"):
        tree = ET.parse(xml)
        changed = False
        for obj in list(tree.getroot().findall("object")):
            if obj.findtext("name") in TEST_LABELS:
                tree.getroot().remove(obj)
                changed = True
        if changed:
            tree.write(xml, encoding="utf-8", xml_declaration=True)

    # COCO: drop test categories from the store, then refresh the JSON.
    coco_dir = TEST_ROOT / "coco-action-test" / "annotations"
    store = coco_dir / ".model_labeling.sqlite3"
    if store.exists():
        import sqlite3 as _sqlite3
        connection = _sqlite3.connect(str(store))
        try:
            rows = connection.execute("SELECT id,payload FROM coco_categories").fetchall()
            dropped = [row[0] for row in rows if json.loads(row[1]).get("name") in TEST_LABELS]
            if dropped:
                marks = ",".join("?" * len(dropped))
                connection.execute(
                    f"DELETE FROM coco_annotations WHERE json_extract(payload,'$.category_id') IN ({marks})",
                    dropped,
                )
                connection.execute(f"DELETE FROM coco_categories WHERE id IN ({marks})", dropped)
                connection.execute("UPDATE coco_document SET dirty=1 WHERE id=1")
                connection.commit()
        finally:
            connection.close()
        if dropped:
            from src.services.annotation_service import AnnotationService
            AnnotationService.export_coco(coco_dir)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._startup_reopen_done = True  # the harness drives dataset opening itself
    window.setGeometry(70, 40, 1780, 1000)
    window.show()
    pump(lambda: window.isVisible(), 5000)
    try:
        return _run_datasets(window)
    finally:
        window.close()
        pump(lambda: False, 1500)


def _run_datasets(window: MainWindow) -> int:
    _cleanup_prior_test_annotations()

    for name, expected_format, expected_task, shapes in DATASETS:
        print(f"\n=== {name} ===", flush=True)
        root = TEST_ROOT / name
        started = time.perf_counter()
        window._start_open_path(root)
        opened = pump(
            lambda: window._dataset_scan_completed and window._dataset_thread is None,
            120000,
        )
        # Let the per-image annotation loader for the selected image settle.
        pump(lambda: window._annotation_thread is None, 30000)
        elapsed = time.perf_counter() - started
        check(opened, f"{name}: dataset opened within {elapsed:.1f}s")
        check(window.settings.annotation_format == expected_format,
              f"{name}: detected format {window.settings.annotation_format} (expect {expected_format})")
        check((window.settings.dataset_task or "") == expected_task,
              f"{name}: dataset task {window.settings.dataset_task} (expect {expected_task})")
        check(window.dataset_total_images > 0, f"{name}: image count {window.dataset_total_images}")
        check(window.state.current_image is not None, f"{name}: an image is selected and displayed")
        task = task_for_format(window.settings.annotation_format, window.settings.dataset_task)
        offered = {ShapeType(window.canvas.method_combo.itemData(index)).value
                   for index in range(window.canvas.method_combo.count())}
        expected = {shape.value for shape in CAPABILITIES[task].shapes}
        check(offered == expected,
              f"{name}: method dropdown offers exactly {sorted(offered)}")
        shot(window, f"{name}_open")

        record = window.state.current_image
        # Wait until the background annotation loader for this image settles,
        # otherwise its reload can overwrite annotations drawn right after open.
        pump(lambda: record.metadata_loaded and window._annotation_thread is None, 30000)
        scene = window.canvas.sceneRect()
        image_w, image_h = scene.width(), scene.height()
        check(image_w > 0 and image_h > 0, f"{name}: canvas has an image to annotate ({int(image_w)}x{int(image_h)})")
        stem, file_name = record.path.stem, record.path.name
        fmt = window.settings.annotation_format
        annotation_dir = Path(window.settings.annotation_dir)
        if fmt == "yolo":
            before_rows, before_classes = yolo_rows(annotation_dir, stem)
        elif fmt == "voc":
            before_count, before_names = voc_rows(annotation_dir, stem)
        else:
            before_count, before_names = coco_rows(annotation_dir, file_name)

        for shape_name, label in shapes:
            shape = ShapeType(shape_name)
            before = len(window.canvas.annotations)
            window.canvas.set_current_label(label)
            enable_shape(window, shape)
            if shape in {ShapeType.RECTANGLE, ShapeType.SQUARE}:
                draw_box(window, shape, QPointF(image_w * 0.18, image_h * 0.18), min(image_w, image_h) * 0.22)
            elif shape == ShapeType.POLYGON:
                draw_polygon(window, QPointF(image_w * 0.45, image_h * 0.30))
            else:
                draw_keypoints(window, QPointF(image_w * 0.55, image_h * 0.55))
            QApplication.processEvents()
            after = window.canvas.annotations
            created = after[before:] if len(after) > before else []
            check(len(created) == 1 and created[0].shape_type == shape,
                  f"{name}: {shape_name} annotation created on canvas")
            if created:
                check(created[0].label == label, f"{name}: {shape_name} uses label '{created[0].label}'")
            check(window.canvas.draw_enabled, f"{name}: method stays armed after {shape_name} (continuous)")
            wait_saved(window)
            check(save_settled(window), f"{name}: {shape_name} auto-saved")

        # Verify the annotation files on disk gained the new rows.
        if fmt == "yolo":
            rows, classes = yolo_rows(annotation_dir, stem)
            check(len(rows) >= len(before_rows) + len(shapes),
                  f"{name}: label file rows {len(before_rows)} -> {len(rows)}")
            for _, label in shapes:
                check(label in classes, f"{name}: classes.txt registers '{label}' (classes={classes})")
        elif fmt == "voc":
            count, names = voc_rows(annotation_dir, stem)
            check(count >= before_count + len(shapes),
                  f"{name}: XML objects {before_count} -> {count}")
            for _, label in shapes:
                check(label in names, f"{name}: XML contains <name>{label}</name>")
        else:
            # The COCO store commits inside the save worker; poll briefly in
            # case the query races the final commit.
            pump(lambda: coco_rows(annotation_dir, file_name)[0] >= before_count + len(shapes), 10000)
            count, labels = coco_rows(annotation_dir, file_name)
            check(count >= before_count + len(shapes),
                  f"{name}: COCO store rows {before_count} -> {count}")
            for _, label in shapes:
                check(label in labels, f"{name}: COCO annotations include label '{label}'")
        shot(window, f"{name}_annotated")

    _run_fresh_dataset_scenario(window)
    _run_obb_scenario(window)
    _finish(window)


def _run_obb_scenario(window: MainWindow) -> None:
    """YOLO OBB: open a rotated-box dataset, draw one, verify the 9-column row."""
    print("\n=== yolo-obb (runtime copy) ===", flush=True)
    obb_root = FRESH_ROOT.parent / "obb-test"
    obb_root.mkdir(parents=True, exist_ok=True)
    (obb_root / "images").mkdir(exist_ok=True)
    (obb_root / "labels").mkdir(exist_ok=True)
    (obb_root / "data.yaml").write_text("task: obb\nnames: [ship]\n", encoding="utf-8")
    for source in sorted(FRESH_SOURCE.glob("*.jpg"))[:2]:
        shutil.copy(source, obb_root / "images" / source.name)

    window._start_open_path(obb_root)
    opened = pump(lambda: window._dataset_scan_completed and window._dataset_thread is None, 120000)
    check(opened, "obb: dataset opened")
    check(window.settings.dataset_task == "yolo_obb", f"obb: task {window.settings.dataset_task}")
    offered = {ShapeType(window.canvas.method_combo.itemData(i)).value for i in range(window.canvas.method_combo.count())}
    check(offered == {"obb"}, f"obb: method dropdown offers only 旋转框 (got {sorted(offered)})")

    record = window.state.current_image
    pump(lambda: record is not None and record.metadata_loaded and window._annotation_thread is None, 30000)
    scene = window.canvas.sceneRect()
    before = len(window.canvas.annotations)
    window.canvas.set_current_label("ship")
    enable_shape(window, ShapeType.OBB)
    draw_box(window, ShapeType.OBB, QPointF(scene.width() * 0.25, scene.height() * 0.25), min(scene.width(), scene.height()) * 0.2)
    QApplication.processEvents()
    created = window.canvas.annotations[before:]
    check(len(created) == 1 and created[0].shape_type == ShapeType.OBB and len(created[0].points) == 4,
          "obb: rotated box drawn with four corners")
    wait_saved(window)
    check(not window.dirty, "obb: auto-saved")

    rows, _ = yolo_rows(obb_root / "labels", record.path.stem)
    check(len(rows) == 1 and len(rows[0].split()) == 9,
          f"obb: label file has one 9-column row (got {len(rows)} rows: {rows[:1]})")
    shot(window, "obb_annotated")
    shutil.rmtree(obb_root.parent, ignore_errors=True)


def _run_fresh_dataset_scenario(window: MainWindow) -> None:
    """A dataset where not a single image is annotated must open and work."""
    print("\n=== fresh-unannotated (runtime copy) ===", flush=True)
    FRESH_ROOT.mkdir(parents=True, exist_ok=True)
    (FRESH_ROOT / "images").mkdir(exist_ok=True)
    (FRESH_ROOT / "labels").mkdir(exist_ok=True)
    for source in sorted(FRESH_SOURCE.glob("*.jpg"))[:3]:
        shutil.copy(source, FRESH_ROOT / "images" / source.name)

    started = time.perf_counter()
    window._start_open_path(FRESH_ROOT)
    opened = pump(
        lambda: window._dataset_scan_completed and window._dataset_thread is None,
        120000,
    )
    elapsed = time.perf_counter() - started
    check(opened, f"fresh: unannotated dataset opened within {elapsed:.1f}s")
    check(window.settings.annotation_format == "yolo" and window.settings.dataset_task == "yolo_detection",
          f"fresh: detected {window.settings.annotation_format}/{window.settings.dataset_task}")
    check(window.dataset_total_images == 3, f"fresh: image count {window.dataset_total_images}")
    record = window.state.current_image
    pump(lambda: record is not None and record.metadata_loaded and window._annotation_thread is None, 30000)
    check(record is not None and len(window.canvas.annotations) == 0,
          "fresh: first image has no annotations and canvas is empty")
    scene = window.canvas.sceneRect()
    check(scene.width() > 0, "fresh: canvas shows the image")

    before = len(window.canvas.annotations)
    window.canvas.set_current_label("person")
    enable_shape(window, ShapeType.RECTANGLE)
    draw_box(window, ShapeType.RECTANGLE, QPointF(scene.width() * 0.2, scene.height() * 0.2), 200)
    QApplication.processEvents()
    created = window.canvas.annotations[before:]
    check(len(created) == 1 and created[0].shape_type == ShapeType.RECTANGLE,
          "fresh: rectangle drawn on the unannotated dataset")
    wait_saved(window)
    check(not window.dirty, "fresh: annotation auto-saved")

    rows, classes = yolo_rows(FRESH_ROOT / "labels", record.path.stem)
    check(len(rows) == 1, f"fresh: label file created with 1 row (got {len(rows)})")
    check("person" in classes, f"fresh: classes.txt auto-created with 'person' (classes={classes})")
    shot(window, "fresh_annotated")
    shutil.rmtree(FRESH_ROOT.parent, ignore_errors=True)


def _finish(window: MainWindow) -> int:
    coco_json = TEST_ROOT / "coco-action-test" / "annotations" / "annotations.json"
    # Let background statistics finish before teardown so closing is clean.
    pump(lambda: window._stats_thread is None, 120000)
    pump(lambda: False, 1500)
    if coco_json.exists():
        document = json.loads(coco_json.read_text(encoding="utf-8"))
        categories = {item["name"] for item in document.get("categories", [])}
        for wanted in ("person", "hand", "foot"):
            check(wanted in categories, f"coco annotations.json exports category '{wanted}'")
    else:
        check(False, "coco annotations.json exported after close")

    print("\n===== SUMMARY =====", flush=True)
    for line in results:
        print(line)
    for message in dialog_errors:
        check(False, f"unexpected dialog: {message[:120]}")
    print(f"total={len(results)} failed={len(failures)}", flush=True)
    code = 1 if failures else 0
    # Qt teardown can segfault after the summary (lingering fetch queues);
    # exit hard so the exit code reflects the test results.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    raise SystemExit(main())
