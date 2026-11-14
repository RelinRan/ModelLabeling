"""Saving a plain image folder converts it to the standard YOLO layout."""
import hashlib
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import yaml
from PIL import Image
from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import src.widgets.main_window as main_window_module
from src.models.annotation import ShapeType
from src.widgets.common_dialogs import AppDialog
from src.widgets.main_window import MainWindow


def _pump(app, predicate, timeout=30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _drop_index_caches(*directories: Path) -> None:
    cache = Path.home() / "AppData" / "Local" / "ModelLabeling" / "index"
    for directory in directories:
        key = hashlib.sha1(str(directory.resolve()).casefold().encode("utf-8")).hexdigest()
        for leftover in cache.glob(f"{key}.sqlite3*"):
            leftover.unlink(missing_ok=True)


def _ultralytics_label_path(image: Path) -> Path:
    resolved = image.resolve().as_posix()
    return Path("/labels/".join(resolved.rsplit("/images/", 1)).rsplit(".", 1)[0] + ".txt")


def test_first_save_migrates_a_plain_folder_and_saves_a_standard_seg_row(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "photos"
    root.mkdir()
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (640, 480), "gray").save(root / name)

    settings_ini = tmp_path / "settings.ini"
    monkeypatch.setattr(
        main_window_module, "QSettings",
        lambda *a, **k: QSettings(str(settings_ini), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr("src.services.label_group_store.DEFAULT_LABEL_GROUP_DB", tmp_path / "label_groups.sqlite3")
    monkeypatch.setattr("src.services.keypoint_group_store.DEFAULT_KEYPOINT_GROUP_DB", tmp_path / "keypoint_groups.sqlite3")
    monkeypatch.setattr(AppDialog, "information", classmethod(lambda cls, *a, **k: None))
    monkeypatch.setattr(AppDialog, "question", classmethod(lambda cls, *a, **k: True))

    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    app.processEvents()
    try:
        window._start_open_path(root)
        assert _pump(app, lambda: window._dataset_scan_completed and window._dataset_thread is None)
        assert _pump(app, lambda: window.state.current_image is not None and window.canvas.image_item is not None)

        # The metadata/annotation load runs in its own thread and reloads the
        # canvas when it lands; drawing before it finishes would be wiped.
        assert _pump(
            app,
            lambda: window.state.current_image is not None
            and window.state.current_image.metadata_loaded
            and window._annotation_thread is None,
        ), "image metadata did not finish loading"
        window.settings.dataset_task = "yolo_segmentation"
        window._apply_annotation_capabilities()
        window._preset_selected("person")
        canvas = window.canvas
        window.activateWindow()
        canvas.setFocus()
        canvas.set_mode(ShapeType.POLYGON)
        canvas._enable_draw_mode()
        app.processEvents()
        assert canvas.mode == ShapeType.POLYGON, (
            canvas.mode, sorted(shape.value for shape in canvas.enabled_shapes)
        )
        assert canvas.draw_enabled

        def point(x, y):
            return canvas.mapFromScene(QPointF(x, y))

        clicks = [(100, 100), (300, 100), (200, 250)]
        mapped = [point(x, y) for x, y in clicks]
        assert all(canvas.viewport().rect().contains(p) for p in mapped), (
            canvas.viewport().size(), canvas.size(), mapped,
        )
        for target in mapped:
            QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, target)
        assert len(canvas.polygon_points) == 3, (len(canvas.polygon_points), canvas.mode, mapped)
        QTest.keyClick(canvas, Qt.Key.Key_Return)
        app.processEvents()
        assert len(canvas.annotations) == 1, [
            (a.shape_type.value, len(a.points)) for a in canvas.annotations
        ]

        images_dir = root / "images"
        assert _pump(app, lambda: (images_dir / "a.jpg").exists() and (images_dir / "b.jpg").exists()), \
            "images were not moved into images/"
        assert not (root / "a.jpg").exists() and not (root / "b.jpg").exists()
        assert _pump(app, lambda: window._save_thread is None), "annotation save did not finish"
        assert window._last_save_error is None, window._last_save_error
        # The layout migration triggers one reload so the session/index follow images/.
        assert _pump(
            app,
            lambda: window.dataset_session is not None
            and window.dataset_session.image_dir.resolve() == images_dir.resolve()
            and window._dataset_scan_completed and window._dataset_thread is None,
        ), "dataset was not reloaded against the standard layout"

        assert window.dataset_root.resolve() == root.resolve()
        assert window.settings.image_dir.resolve() == images_dir.resolve()
        assert window.settings.annotation_dir.resolve() == (root / "labels").resolve()

        labels = sorted((root / "labels").glob("*.txt"))
        assert labels, "no label file was written"
        saved = [label for label in labels if label.read_text(encoding="utf-8").strip()]
        assert len(saved) == 1, [(p.name, p.read_text(encoding="utf-8")) for p in labels]
        row = saved[0].read_text(encoding="utf-8").split()
        assert len(row) == 7, row  # class + three normalized points
        assert row[0] == "0", row
        assert all(0.0 <= float(value) <= 1.0 for value in row[1:]), row
        # The trainer's own rule now maps the image to the saved label.
        assert _ultralytics_label_path(images_dir / "a.jpg") == saved[0].resolve()

        document = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
        assert document["task"] == "segment"
        assert document["path"] == root.resolve().as_posix()
        assert document["train"] == "images" and document["val"] == "images"
        # The app's preset table is the dataset's class table; the drawn label
        # keeps the class id the saved row uses.
        names = document["names"]
        assert sorted(names) == list(range(len(names)))
        assert names[0] == "person"
        assert (root / "classes.txt").read_text(encoding="utf-8").split("\n")[: len(names)] == [
            names.get(index) for index in range(len(names))
        ]
        assert window.state.current_image is not None
        assert str(window.state.current_image.path).casefold().startswith(str(images_dir).casefold())
    finally:
        window.close()
        _drop_index_caches(root, root / "images")