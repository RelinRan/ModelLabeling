import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QPointF, QSettings
from PySide6.QtWidgets import QApplication
import pytest

import src.widgets.main_window as main_window_module
from src.models.annotation import Annotation, LabelPreset, ShapeType
from src.models.project import ProjectSettings
from src.services.annotation_service import AnnotationService
from src.widgets.cleanup_dialog import CleanupDialog
from src.widgets.main_window import MainWindow


def _wait(app, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _make_yolo(root: Path, names=("a.jpg", "b.jpg", "c.jpg")):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("person\ncar\n", encoding="utf-8")
    for name in names:
        Image.new("RGB", (200, 140), (60, 100, 140)).save(root / "images" / name, "JPEG")
        (root / "labels" / Path(name).with_suffix(".txt")).write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )


def _make_second_dataset(root: Path):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.new("RGB", (200, 140), (120, 80, 60)).save(root / "images" / "other.jpg", "JPEG")
    (root / "labels" / "other.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "classes.txt").write_text("person\n", encoding="utf-8")


def _open(window, app, root: Path):
    window._start_open_path(root)
    assert _wait(app, lambda: window.dataset_root is not None and window.dataset_root.resolve() == root.resolve())
    assert _wait(app, lambda: window._dataset_scan_completed and window._dataset_thread is None)
    assert _wait(app, lambda: window.state.current_image is not None)


def _select(window, app, name: str):
    row = next(index for index, item in enumerate(window.image_panel.records) if item.path.name == name)
    window.image_panel.list.setCurrentRow(row)
    assert _wait(app, lambda: window.state.current_image is not None and window.state.current_image.path.name == name)
    assert _wait(app, lambda: window._annotation_thread is None and window.state.current_image.metadata_loaded, 60)
    return window.state.current_image


def _save_and_wait(window, app):
    window.save_current()
    assert _wait(app, lambda: window._save_thread is None or not window._save_thread.is_alive())
    assert _wait(app, lambda: not window.dirty)


def _add_box(window, label="car", x=20, y=20):
    annotation = Annotation(
        ShapeType.RECTANGLE, label,
        [QPointF(x, y), QPointF(x + 50, y + 40)],
        color="#ffcc00",
    )
    window.canvas.annotations.append(annotation)
    window.canvas._add_annotation_item(annotation)
    window.canvas.annotationCreated.emit(annotation)
    window.canvas.dirtyChanged.emit(True)


def test_generated_dataset_multi_image_edit_cleanup_history_reload_compare(tmp_path, monkeypatch, run_cleanup):
    """Generate fresh data and verify edits survive cleanup, history, and reload."""
    app = QApplication.instance() or QApplication([])
    settings_ini = tmp_path / "settings.ini"
    monkeypatch.setattr(main_window_module, "QSettings", lambda *a, **k: QSettings(str(settings_ini), QSettings.Format.IniFormat))
    monkeypatch.setattr("src.widgets.main_window.AppDialog.information", classmethod(lambda cls, *a, **k: None))
    monkeypatch.setattr("src.widgets.cleanup_dialog.AppDialog.information", classmethod(lambda cls, *a, **k: None))
    monkeypatch.setattr("src.widgets.cleanup_dialog.AppDialog.question", classmethod(lambda cls, *a, **k: True))

    dataset = tmp_path / "dataset-a"
    other = tmp_path / "dataset-b"
    _make_yolo(dataset)
    _make_second_dataset(other)
    window = MainWindow()
    try:
        _open(window, app, dataset)
        assert len(window.state.images) == 3

        # a.jpg: modify the existing label and add a second box.
        _select(window, app, "a.jpg")
        window.canvas.annotation_items[0].setSelected(True)
        assert window.canvas.update_selected_label("car", "#ffcc00")
        _add_box(window, "person", 100, 50)
        _save_and_wait(window, app)

        # b.jpg: delete the original box and add a replacement box.
        _select(window, app, "b.jpg")
        window.canvas.annotation_items[0].setSelected(True)
        assert window.canvas.delete_selected()
        _add_box(window, "car", 30, 25)
        _save_and_wait(window, app)

        # c.jpg: delete the only box; cleanup must remove this image and file.
        _select(window, app, "c.jpg")
        window.canvas.annotation_items[0].setSelected(True)
        assert window.canvas.delete_selected()
        _save_and_wait(window, app)
        assert (dataset / "labels" / "c.txt").read_text(encoding="utf-8") == ""

        # Compare the saved files before cleanup.
        service = AnnotationService()
        settings = ProjectSettings(
            image_dir=dataset / "images", annotation_dir=dataset / "labels",
            annotation_format="yolo", dataset_task="yolo_detection",
            label_presets=[LabelPreset("person", 0, "#00e5ff"), LabelPreset("car", 1, "#ffcc00")],
        )
        loaded_a = service.load(dataset / "images" / "a.jpg", dataset / "labels", settings)
        loaded_b = service.load(dataset / "images" / "b.jpg", dataset / "labels", settings)
        assert [item.label for item in loaded_a.annotations] == ["car", "person"]
        assert [item.label for item in loaded_b.annotations] == ["car"]

        # Run the real cleanup dialog on the generated dataset.
        dialog = CleanupDialog(window.settings.label_presets, window, str(dataset), window.settings.language)
        dialog._start_scan()
        assert _wait(app, lambda: not dialog._scanning)
        # c.jpg was cleared of its box above, so it is a box-less image and
        # its removal is the quality rule the user has to switch on.
        run_cleanup(dialog, quality=True)
        assert not (dataset / "images" / "c.jpg").exists()
        assert not (dataset / "labels" / "c.txt").exists()
        dialog.close()

        # Switch by the persisted history paths, then return and reload.
        assert str(dataset.resolve()) in window._history_paths()
        _open(window, app, other)
        assert str(other.resolve()) in window._history_paths()
        _open(window, app, dataset)
        assert _wait(app, lambda: window.state.current_image is not None and window.state.current_image.metadata_loaded)
        assert sorted(item.path.name for item in window.state.images) == ["a.jpg", "b.jpg"]

        # Reloaded canvas/file contents must match the expected edits.
        _select(window, app, "a.jpg")
        assert [item.label for item in window.canvas.annotations] == ["car", "person"]
        _select(window, app, "b.jpg")
        assert [item.label for item in window.canvas.annotations] == ["car"]
        assert window.canvas.annotations[0].points[0].x() == pytest.approx(30, abs=0.01)
    finally:
        window.close()
