# -*- coding: utf-8 -*-
"""Guards for the conftest isolation and the synthetic-dataset factory.

The point of these tests is that the *other* tests can be trusted: they prove
tests cannot reach the developer's real settings or real dataset, and that the
synthetic builder actually produces the defects asked for.
"""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from src.services.dataset_detector import DatasetDetector
from src.widgets import main_window as main_window_module
from src.widgets.main_window import MainWindow

from conftest import VOC_TEST_ROOT, VocTestDataset, dataset_guard


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _real_registry_value(key: str) -> str:
    store = QSettings(QSettings.Format.NativeFormat, QSettings.Scope.UserScope,
                      "RelinRan", "ModelLabeling")
    return str(store.value(key, "") or "")


def test_isolated_settings_never_read_the_real_store(tmp_path):
    """The redirected store is a temp INI, not the developer's registry."""
    store = main_window_module.QSettings()
    stored = Path(store.fileName())
    assert stored.suffix == ".ini"
    assert stored.parent == tmp_path
    # The real store may well have a last dataset; the test store must not.
    assert store.value("reopen/last_root", "") in ("", None)


def test_main_window_does_not_open_the_developers_dataset(synthetic_dataset):
    """A stray MainWindow must not adopt whatever dataset is in the registry.

    This is the regression that made a shortcut probe open a real dataset:
    showEvent -> _maybe_reopen_last_dataset reads reopen/last_root.
    """
    real_root = _real_registry_value("reopen/last_root")
    dataset = synthetic_dataset("scratch", fmt="voc", frames=3)

    app = _app()
    window = MainWindow()
    window.show()
    app.processEvents()

    opened = str(window.dataset_root) if window.dataset_root else ""
    assert opened != real_root, "MainWindow opened the developer's real dataset"
    assert "scratch" not in opened
    assert opened == ""

    window.close()
    # Reading and closing must not have rewritten the developer's registry.
    assert _real_registry_value("reopen/last_root") == real_root


def test_synthetic_voc_dataset_is_detected(synthetic_dataset):
    root = synthetic_dataset("voc-basic", fmt="voc", frames=4, classes=("4Ra",))
    detected = DatasetDetector.detect(root)
    assert detected.format_name == "voc"
    assert detected.image_dir == root / "JPEGImages"
    assert detected.annotation_dir == root / "Annotations"
    assert len(list((root / "JPEGImages").glob("*.jpg"))) == 4
    assert len(list((root / "Annotations").glob("*.xml"))) == 4
    with Image.open(next((root / "JPEGImages").glob("*.jpg"))) as image:
        assert image.size == (640, 480)


def test_synthetic_yolo_dataset_is_detected(synthetic_dataset):
    root = synthetic_dataset("yolo-basic", fmt="yolo", frames=3)
    detected = DatasetDetector.detect(root)
    assert detected.format_name == "yolo"
    assert len(list((root / "labels" / "train").glob("*.txt"))) == 3


def test_synthetic_coco_dataset_is_detected(synthetic_dataset):
    root = synthetic_dataset("coco-basic", fmt="coco", frames=3)
    detected = DatasetDetector.detect(root)
    assert detected.format_name == "coco"


def test_defects_are_actually_injected(synthetic_dataset):
    """Each requested defect must land in the data, or the fixture lies."""
    import xml.etree.ElementTree as ET

    root = synthetic_dataset("defects", fmt="voc", frames=12, classes=("4R", "4Ra"),
                             defects=("no_object", "small_box", "zero_area", "duplicate_box",
                                      "near_duplicate_box", "label_conflict",
                                      "out_of_bounds", "wrong_depth"))
    ann = root / "Annotations"
    parsed = {}
    for path in sorted(ann.glob("*.xml")):
        root_el = ET.parse(path).getroot()
        boxes = []
        for obj in root_el.iter("object"):
            bb = obj.find("bndbox")
            boxes.append((
                obj.findtext("name"),
                float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                float(bb.findtext("xmax")), float(bb.findtext("ymax")),
            ))
        parsed[path.stem] = (root_el, boxes)

    # no_object: some frames carry no annotation file at all
    assert len(parsed) < 12

    all_boxes = [b for v in parsed.values() for b in v[1]]
    sizes = [(b[3] - b[1], b[4] - b[2]) for b in all_boxes]

    assert any(w < 20 and h < 20 for w, h in sizes), "small_box not injected"
    # zero_area: wide but zero-height, so the "w < 20 and h < 20" rule misses it
    assert any(w > 20 and h == 0 for w, h in sizes), "zero_area not injected"
    assert any(b[1] < 0 or b[2] < 0 for b in all_boxes), "out_of_bounds not injected"

    # duplicate / near-duplicate / conflict appear on the same images
    multi = [boxes for boxes in (v[1] for v in parsed.values()) if len(boxes) > 1]
    assert any(boxes[0][1:] == boxes[1][1:] for boxes in multi), "duplicate_box not injected"
    assert any(boxes[0][0] != boxes[1][0] for boxes in multi), "label_conflict not injected"

    depths = {v[0].findtext("size/depth") for v in parsed.values()}
    assert "4" in depths, "wrong_depth not injected"


def test_size_mismatch_and_unreadable_image(synthetic_dataset):
    import xml.etree.ElementTree as ET

    root = synthetic_dataset("bad-images", fmt="voc", frames=6,
                             defects=("size_mismatch", "unreadable_image"))
    mismatched = 0
    unreadable = 0
    for path in (root / "Annotations").glob("*.xml"):
        declared = ET.parse(path).getroot().find("size")
        image_path = root / "JPEGImages" / f"{path.stem}.jpg"
        try:
            with Image.open(image_path) as image:
                # A truncated JPEG keeps a readable header; only a real decode
                # exposes it.
                image.load()
                if image.size != (int(declared.findtext("width")), int(declared.findtext("height"))):
                    mismatched += 1
        except OSError:
            unreadable += 1
    assert mismatched > 0, "size_mismatch not injected"
    assert unreadable > 0, "unreadable_image not injected"


def test_orphan_annotation(synthetic_dataset):
    root = synthetic_dataset("orphans", fmt="voc", frames=3, defects=("orphan_annotation",))
    stems = {p.stem for p in (root / "Annotations").glob("*.xml")}
    images = {p.stem for p in (root / "JPEGImages").glob("*.jpg")}
    assert stems - images, "orphan_annotation not injected"


# --------------------------------------------------------------------------
# the designated test dataset
# --------------------------------------------------------------------------


def test_voc_test_dataset_is_the_designated_one(voc_test):
    assert voc_test.root == VOC_TEST_ROOT
    assert voc_test.image_dir == VOC_TEST_ROOT / "JPEGImages"
    assert voc_test.annotation_dir == VOC_TEST_ROOT / "Annotations"
    # It is a real dataset, so sanity-check the shape rather than a fixture.
    assert len(voc_test.images()) > 0
    assert len(voc_test.images()) == len(voc_test.annotations())


def test_voc_test_stage_isolates_writes(voc_test, tmp_path):
    """Staged copies are what mutating tests work on."""
    staged = voc_test.stage("slice", limit=5)
    assert str(staged).startswith(str(tmp_path))
    assert len(list((staged / "JPEGImages").glob("*.jpg"))) == 5
    assert len(list((staged / "Annotations").glob("*.xml"))) == 5

    # Mutating the copy must not touch the shared dataset.
    before = voc_test.fingerprint()
    target = next((staged / "Annotations").glob("*.xml"))
    target.write_text("<annotation/>", encoding="utf-8")
    (staged / "JPEGImages" / "not-a-real-image.jpg").write_bytes(b"x")
    assert voc_test.fingerprint() == before


def test_dataset_guard_detects_in_place_edits(voc_test, tmp_path):
    """The guard must actually fail when its dataset is written to.

    Exercised against a staged copy so the shared dataset is never at risk.
    """
    staged = VocTestDataset(voc_test.stage("guarded", limit=3), tmp_path)

    with pytest.raises(AssertionError, match="modified the shared test dataset"):
        with dataset_guard(staged):
            next((staged.root / "Annotations").glob("*.xml")).write_text(
                "<annotation/>", encoding="utf-8")

    # An untouched dataset passes the same guard.
    with dataset_guard(VocTestDataset(voc_test.stage("clean", limit=3), tmp_path)):
        pass


def test_dataset_guard_ignores_reads(voc_test):
    """Reading the dataset must not trip the guard."""
    with dataset_guard(voc_test):
        for path in voc_test.images()[:3]:
            path.read_bytes()
        for path in voc_test.annotations()[:3]:
            path.read_text(encoding="utf-8")
