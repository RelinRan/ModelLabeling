import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import src.widgets.main_window as main_window_module
from src.widgets.main_window import MainWindow


def _wait(app, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 120), (80, 120, 160)).save(path, "JPEG")


def _make_yolo(root):
    image = root / "images" / "a" / "frame.jpg"
    _image(image)
    (root / "labels" / "a").mkdir(parents=True)
    (root / "labels" / "a" / "frame.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    (root / "classes.txt").write_text("person\n", encoding="utf-8")
    return image, root / "labels" / "a" / "frame.txt"


def _make_voc(root):
    image = root / "JPEGImages" / "frame.jpg"
    _image(image)
    (root / "Annotations").mkdir(parents=True)
    (root / "Annotations" / "frame.xml").write_text(
        "<annotation><filename>frame.jpg</filename>"
        "<size><width>160</width><height>120</height><depth>3</depth></size>"
        "<object><name>person</name><bndbox><xmin>20</xmin><ymin>20</ymin>"
        "<xmax>80</xmax><ymax>80</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )
    return image, root / "Annotations" / "frame.xml"


def _make_coco(root):
    image = root / "images" / "frame.jpg"
    _image(image)
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "annotations.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "frame.jpg", "width": 160, "height": 120}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [20, 20, 60, 60], "area": 3600, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "person"}],
    }), encoding="utf-8")
    return image, annotations


def test_generated_gui_open_delete_save_switch_reload_all_formats(tmp_path, monkeypatch):
    """Use only freshly generated datasets for the end-to-end GUI path."""
    app = QApplication.instance() or QApplication([])
    settings_ini = tmp_path / "settings.ini"
    monkeypatch.setattr(main_window_module, "QSettings", lambda *a, **k: QSettings(str(settings_ini), QSettings.Format.IniFormat))
    monkeypatch.setattr("src.widgets.main_window.AppDialog.information", classmethod(lambda cls, *a, **k: None))
    monkeypatch.setattr("src.widgets.main_window.AppDialog.question", classmethod(lambda cls, *a, **k: True))
    cases = [
        (tmp_path / "yolo", _make_yolo(tmp_path / "yolo")),
        (tmp_path / "voc", _make_voc(tmp_path / "voc")),
        (tmp_path / "coco", _make_coco(tmp_path / "coco")),
    ]
    window = MainWindow()
    try:
        for root, (image, annotation_path) in cases:
            window._start_open_path(root)
            assert _wait(app, lambda: window._dataset_scan_completed and window._dataset_thread is None)
            assert _wait(app, lambda: window.state.current_image is not None and window.canvas.annotations)
            assert window.state.current_image.path.resolve() == image.resolve()
            assert len(window.canvas.annotations) == 1
            window.canvas.annotation_items[0].setSelected(True)
            window.canvas.delete_selected()
            assert _wait(app, lambda: window._save_thread is None or not window._save_thread.is_alive())
            assert _wait(app, lambda: not window.dirty)
            # Switching away and back is the regression scenario: the deleted
            # annotation must not be loaded from stale memory or disk.
            other_root = tmp_path / ("other-" + root.name)
            other_image = other_root / "images" / "other.jpg"
            _image(other_image)
            window._start_open_path(other_root)
            assert _wait(app, lambda: window._dataset_scan_completed and window._dataset_thread is None)
            window._start_open_path(root)
            assert _wait(app, lambda: window._dataset_scan_completed and window._dataset_thread is None)
            assert _wait(app, lambda: window.state.current_image is not None and window.state.current_image.metadata_loaded)
            assert window.canvas.annotations == []
            if root.name != "coco":
                assert annotation_path.exists()
        # Ensure the final generated dataset was actually deleted on disk.
        assert not (tmp_path / "yolo" / "labels" / "a" / "frame.txt").read_text(encoding="utf-8").strip()
        assert not (tmp_path / "voc" / "Annotations" / "frame.xml").read_text(encoding="utf-8").count("<object>")
    finally:
        window.close()
