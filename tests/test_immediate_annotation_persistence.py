from pathlib import Path
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import src.widgets.main_window as main_window_module
from src.widgets.main_window import MainWindow


def _wait(app, predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_annotation_mutation_is_persisted_on_next_event_loop_turn(tmp_path, monkeypatch):
    """Create/delete/rename edits must reach disk without an explicit Save."""
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(
        main_window_module,
        "QSettings",
        lambda *args, **kwargs: QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat),
    )
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("person\ncar\n", encoding="utf-8")
    Image.new("RGB", (160, 100), (80, 100, 120)).save(root / "images" / "sample.jpg", "JPEG")
    (root / "labels" / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    window = MainWindow()
    try:
        window._start_open_path(root)
        assert _wait(app, lambda: window.state.current_image is not None and window._dataset_scan_completed)
        assert _wait(app, lambda: window._annotation_thread is None and window.state.current_image.metadata_loaded)

        item = window.canvas.annotation_items[0]
        item.setSelected(True)
        assert window.canvas.update_selected_label("car", "#ffcc00")
        assert _wait(app, lambda: window._save_worker is None and not window.dirty)
        assert (root / "labels" / "sample.txt").read_text(encoding="utf-8").split()[0] == "1"
        assert window.state.current_image.annotations[0].label == "car"

        window.canvas.delete_selected()
        assert _wait(app, lambda: window._save_worker is None and not window.dirty)
        assert (root / "labels" / "sample.txt").read_text(encoding="utf-8") == ""
        assert window.state.current_image.annotations == []
    finally:
        window.close()
