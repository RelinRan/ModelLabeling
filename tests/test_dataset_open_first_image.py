import hashlib
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from PIL import Image

import src.widgets.main_window as main_window_module
from src.widgets.main_window import MainWindow


def _make_dataset(root: Path) -> Path:
    """A YOLO dataset whose walk order differs from its sorted order.

    os.walk visits the root's files (a.jpg, z.jpg) before the subfolder,
    while the image list sorts by relative path ("0sub/x.jpg" first). The
    preview must follow the sorted list, not the scan order.
    """
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for relative, annotated in (("0sub/x.jpg", True), ("a.jpg", True), ("z.jpg", False)):
        image_path = root / "images" / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (200, 10, 10)).save(image_path, "JPEG")
        label_relative = str(Path(relative).with_suffix(".txt")).replace("\\", "/")
        label_path = root / "labels" / label_relative
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("0 0.5 0.5 0.2 0.2\n" if annotated else "", encoding="utf-8")
    return root


def _pump_until_loaded(app, window, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if window._dataset_scan_completed and window._dataset_load_succeeded:
            # Let the deferred finish handler (QTimer.singleShot(0)) run.
            for _ in range(20):
                app.processEvents()
                time.sleep(0.01)
            return True
        time.sleep(0.01)
    return False


def _pump_until(app, condition, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def _drop_index_cache(root: Path) -> None:
    """Remove the per-dataset index cache this open created."""
    key = hashlib.sha1(str(root.resolve()).casefold().encode("utf-8")).hexdigest()
    cache = Path.home() / "AppData" / "Local" / "ModelLabeling" / "index"
    for leftover in cache.glob(f"{key}.sqlite3*"):
        leftover.unlink(missing_ok=True)


def test_opening_a_dataset_shows_its_first_image(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    root = _make_dataset(tmp_path / "yolo")
    # Keep the test hermetic: settings INI and group stores live in tmp_path.
    settings_ini = tmp_path / "settings.ini"
    monkeypatch.setattr(
        main_window_module, "QSettings",
        lambda *a, **k: QSettings(str(settings_ini), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr("src.services.label_group_store.DEFAULT_LABEL_GROUP_DB", tmp_path / "label_groups.sqlite3")
    monkeypatch.setattr("src.services.keypoint_group_store.DEFAULT_KEYPOINT_GROUP_DB", tmp_path / "keypoint_groups.sqlite3")

    window = MainWindow()
    try:
        window._start_open_path(root)
        assert _pump_until_loaded(app, window), "dataset scan did not finish"

        first = root / "images" / "0sub" / "x.jpg"
        assert window.state.current_image is not None
        assert str(window.state.current_image.path).casefold() == str(first).casefold()
        assert window.image_panel.list.currentRow() == 0
        assert str(window.image_panel.records[0].path).casefold() == str(first).casefold()
    finally:
        window.close()
        _drop_index_cache(root)


def test_switching_datasets_clears_the_preview(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    first = _make_dataset(tmp_path / "one")
    second = _make_dataset(tmp_path / "two")
    settings_ini = tmp_path / "settings.ini"
    monkeypatch.setattr(
        main_window_module, "QSettings",
        lambda *a, **k: QSettings(str(settings_ini), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr("src.services.label_group_store.DEFAULT_LABEL_GROUP_DB", tmp_path / "label_groups.sqlite3")
    monkeypatch.setattr("src.services.keypoint_group_store.DEFAULT_KEYPOINT_GROUP_DB", tmp_path / "keypoint_groups.sqlite3")

    window = MainWindow()
    try:
        window._start_open_path(first)
        assert _pump_until_loaded(app, window), "first dataset scan did not finish"
        assert _pump_until(app, lambda: window.canvas.image_item is not None)
        assert window.canvas.annotations

        window._start_open_path(second)
        # Without any event pumping the old image must already be gone: the
        # preview never keeps painting the previous dataset's picture.
        assert window.canvas.image_item is None
        assert window.canvas.annotation_items == []
        assert window.canvas.annotations == []

        assert _pump_until_loaded(app, window), "second dataset scan did not finish"
        assert _pump_until(app, lambda: window.canvas.image_item is not None)
        assert str(window.state.current_image.path).casefold().startswith(str(second.resolve()).casefold())
    finally:
        window.close()
        _drop_index_cache(first)
        _drop_index_cache(second)
