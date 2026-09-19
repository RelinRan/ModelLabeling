import hashlib
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from PIL import Image

import src.widgets.main_window as main_window_module
from src.widgets.cleanup_dialog import CleanupDialog
from src.widgets.main_window import MainWindow

BOX = "0 0.5 0.5 0.2 0.2\n"
INDEX_CACHE = Path.home() / "AppData" / "Local" / "ModelLabeling" / "index"


def _make_dataset(root: Path, layout: dict) -> Path:
    root.mkdir(parents=True)
    (root / "images").mkdir()
    (root / "labels").mkdir()
    for relative, rows in layout.items():
        image_path = root / "images" / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (200, 10, 10)).save(image_path, "JPEG")
        label_path = root / "labels" / (Path(relative).with_suffix(".txt").as_posix())
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(rows, encoding="utf-8")
    return root


def _pump_until(app, condition, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            for _ in range(20):
                app.processEvents()
                time.sleep(0.01)
            return True
        time.sleep(0.01)
    return False


def _drop_index_cache(root: Path) -> None:
    key = hashlib.sha1(str(root.resolve()).casefold().encode("utf-8")).hexdigest()
    for leftover in INDEX_CACHE.glob(f"{key}.sqlite3*"):
        leftover.unlink(missing_ok=True)


def test_cleanup_after_clearing_last_box_reloads_dataset(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    # a.jpg keeps its box; the user will clear it inside the app.
    dataset = _make_dataset(tmp_path / "yolo", {"a.jpg": BOX, "b.jpg": BOX})
    other = _make_dataset(tmp_path / "other", {"x.jpg": BOX})
    settings_ini = tmp_path / "settings.ini"
    monkeypatch.setattr(
        main_window_module, "QSettings",
        lambda *a, **k: QSettings(str(settings_ini), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr("src.services.label_group_store.DEFAULT_LABEL_GROUP_DB", tmp_path / "label_groups.sqlite3")
    monkeypatch.setattr("src.services.keypoint_group_store.DEFAULT_KEYPOINT_GROUP_DB", tmp_path / "keypoint_groups.sqlite3")
    monkeypatch.setattr(
        "src.widgets.cleanup_dialog.AppDialog.information",
        classmethod(lambda cls, *a, **k: None),
    )

    window = MainWindow()
    try:
        # 1) open the dataset; a.jpg (first) loads with its box.
        window._start_open_path(dataset)
        assert _pump_until(app, lambda: window._dataset_scan_completed and window._dataset_load_succeeded)
        assert _pump_until(app, lambda: window.state.current_image is not None and window.canvas.annotation_items)

        # 2) the user clears the only box; the autosave timer is pending
        #    (no event loop time passes), so the edit is still unsaved.
        window.canvas.annotation_items[0].setSelected(True)
        window.canvas.delete_selected()
        assert window.dirty
        assert (dataset / "labels" / "a.txt").read_text(encoding="utf-8") == BOX

        # 3) opening the cleanup dialog flushes the edit to disk first.
        window._flush_pending_annotation_save()
        assert (dataset / "labels" / "a.txt").read_text(encoding="utf-8") == ""

        # 4) scan + clean through the real dialog logic.
        dialog = CleanupDialog([], window, str(dataset))
        monkeypatch.setattr(dialog, "_trash", lambda path: (path.unlink(missing_ok=True), True)[1])
        dialog._start_scan()
        assert _pump_until(app, lambda: not dialog._scanning)
        report = dialog.result_label.toPlainText().splitlines()
        assert report[0] == "[总图片]：2张  [有效标注]：1张  [无标注]：1张"
        assert "[无标注] a.jpg" in report
        dialog._clean()
        assert not (dataset / "images" / "a.jpg").exists()
        assert not (dataset / "labels" / "a.txt").exists()

        # 5) the main window reloads the cleaned dataset instead of keeping
        #    the stale open one.
        window._reload_cleaned_dataset(dataset)
        assert _pump_until(app, lambda: window._dataset_scan_completed and window._dataset_load_succeeded)
        listed = [record.path.name for record in window.state.images]
        assert listed == ["b.jpg"]
        assert window.state.current_image is not None
        assert window.state.current_image.path.name == "b.jpg"

        # 6) switching away and back keeps it gone.
        window._start_open_path(other)
        assert _pump_until(app, lambda: window._dataset_scan_completed and window._dataset_load_succeeded)
        window._start_open_path(dataset)
        assert _pump_until(app, lambda: window._dataset_scan_completed and window._dataset_load_succeeded)
        assert [record.path.name for record in window.state.images] == ["b.jpg"]
    finally:
        window.close()
        _drop_index_cache(dataset)
        _drop_index_cache(other)
