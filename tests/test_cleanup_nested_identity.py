import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication

from src.widgets.cleanup_dialog import CleanupDialog


def _wait(app, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _image(path: Path, color=(80, 100, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color).save(path, "JPEG")


def test_coco_cleanup_uses_relative_path_for_duplicate_basenames(tmp_path, run_cleanup):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "coco-nested"
    _image(root / "images" / "train" / "shared.jpg", (120, 40, 40))
    _image(root / "images" / "val" / "shared.jpg", (40, 120, 40))
    _image(root / "images" / "val" / "keep.jpg", (40, 40, 120))
    (root / "annotations").mkdir(parents=True)
    document = {
        "images": [
            {"id": 1, "file_name": "train/shared.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "val/shared.jpg", "width": 100, "height": 80},
            {"id": 3, "file_name": "val/keep.jpg", "width": 100, "height": 80},
        ],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 3, "category_id": 1, "bbox": [30, 20, 20, 20], "area": 400, "iscrowd": 0},
        ],
    }
    json_path = root / "annotations" / "annotations.json"
    json_path.write_text(json.dumps(document), encoding="utf-8")

    dialog = CleanupDialog([], default_source=str(root))
    try:
        dialog._start_scan()
        assert _wait(app, lambda: not dialog._scanning)
        assert [path.relative_to(root).as_posix() for path in dialog._to_delete_images] == ["images/val/shared.jpg"]
        # The record with no annotations is a quality call: it is the switch
        # a user turns on when they want empty images gone.
        run_cleanup(dialog, quality=True)

        assert (root / "images" / "train" / "shared.jpg").exists()
        assert not (root / "images" / "val" / "shared.jpg").exists()
        assert (root / "images" / "val" / "keep.jpg").exists()
        cleaned = json.loads(json_path.read_text(encoding="utf-8"))
        assert [item["file_name"] for item in cleaned["images"]] == ["train/shared.jpg", "val/keep.jpg"]
        assert {item["image_id"] for item in cleaned["annotations"]} == {1, 3}
    finally:
        dialog.close()


def test_yolo_cleanup_detects_nested_orphan_with_duplicate_stem(tmp_path, run_cleanup):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "yolo-nested"
    _image(root / "images" / "train" / "shared.jpg")
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "labels" / "train" / "shared.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    orphan = root / "labels" / "val" / "shared.txt"
    orphan.write_text("0 0.25 0.25 0.1 0.1\n", encoding="utf-8")
    (root / "classes.txt").write_text("person\n", encoding="utf-8")

    dialog = CleanupDialog([], default_source=str(root))
    try:
        dialog._start_scan()
        assert _wait(app, lambda: not dialog._scanning)
        assert dialog._to_delete_images == []
        assert [path.relative_to(root).as_posix() for path in dialog._to_delete_annotations] == ["labels/val/shared.txt"]
        # An orphan is a broken invariant, not a policy call, so the default
        # structural policy is enough to act on it.
        run_cleanup(dialog)

        assert (root / "labels" / "train" / "shared.txt").exists()
        assert not orphan.exists()
    finally:
        dialog.close()



def test_coco_cleanup_keeps_json_record_when_image_delete_fails(tmp_path, run_cleanup, locked_file):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "coco-delete-failure"
    image_path = root / "images" / "nested" / "empty.jpg"
    _image(image_path)
    (root / "annotations").mkdir(parents=True)
    json_path = root / "annotations" / "annotations.json"
    document = {
        "images": [{"id": 7, "file_name": "nested/empty.jpg", "width": 100, "height": 80}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [],
    }
    json_path.write_text(json.dumps(document), encoding="utf-8")

    dialog = CleanupDialog([], default_source=str(root))
    try:
        dialog._start_scan()
        assert _wait(app, lambda: not dialog._scanning)
        assert dialog._to_delete_images == [image_path]
        with locked_file(image_path):
            lines = run_cleanup(dialog, quality=True)

        assert any("未能删除" in line for line in lines), lines
        assert image_path.exists()
        cleaned = json.loads(json_path.read_text(encoding="utf-8"))
        assert cleaned["images"] == document["images"]
        assert cleaned["annotations"] == []
    finally:
        dialog.close()


def test_yolo_cleanup_keeps_empty_label_when_image_delete_fails(tmp_path, run_cleanup, locked_file):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "yolo-delete-failure"
    image_path = root / "images" / "nested" / "empty.jpg"
    label_path = root / "labels" / "nested" / "empty.txt"
    _image(image_path)
    label_path.parent.mkdir(parents=True)
    label_path.write_text("", encoding="utf-8")
    (root / "classes.txt").write_text("person\n", encoding="utf-8")

    dialog = CleanupDialog([], default_source=str(root))
    try:
        dialog._start_scan()
        assert _wait(app, lambda: not dialog._scanning)
        assert dialog._to_delete_images == [image_path]
        with locked_file(image_path):
            lines = run_cleanup(dialog, quality=True)

        assert any("未能删除" in line for line in lines), lines
        # And it must not be reported as cleaned: the log is where a user
        # checks what a run did, and the file is still there.
        assert not any(line.startswith("[已清理]") for line in lines), lines
        assert image_path.exists()
        assert label_path.exists()
        assert label_path.read_text(encoding="utf-8") == ""
    finally:
        dialog.close()


def _voc_xml(filename: str, objects: list[tuple[str, tuple[int, int, int, int]]]) -> str:
    rows = [
        "<annotation>",
        f"  <filename>{filename}</filename>",
        "  <size><width>100</width><height>80</height><depth>3</depth></size>",
    ]
    for label, (xmin, ymin, xmax, ymax) in objects:
        rows.extend([
            "  <object>",
            f"    <name>{label}</name>",
            "    <bndbox>",
            f"      <xmin>{xmin}</xmin><ymin>{ymin}</ymin>",
            f"      <xmax>{xmax}</xmax><ymax>{ymax}</ymax>",
            "    </bndbox>",
            "  </object>",
        ])
    rows.append("</annotation>")
    return "\n".join(rows)


def test_voc_cleanup_uses_nested_relative_identity_and_keeps_pair_on_failure(
    tmp_path, run_cleanup, locked_file,
):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "voc-nested"
    keep_image = root / "JPEGImages" / "train" / "shared.jpg"
    empty_image = root / "JPEGImages" / "val" / "shared.jpg"
    _image(keep_image, (150, 40, 40))
    _image(empty_image, (40, 150, 40))
    keep_xml = root / "Annotations" / "train" / "shared.xml"
    empty_xml = root / "Annotations" / "val" / "shared.xml"
    orphan_xml = root / "Annotations" / "orphan" / "shared.xml"
    for path in (keep_xml, empty_xml, orphan_xml):
        path.parent.mkdir(parents=True, exist_ok=True)
    keep_xml.write_text(_voc_xml("shared.jpg", [("person", (10, 10, 30, 30))]), encoding="utf-8")
    empty_xml.write_text(_voc_xml("shared.jpg", []), encoding="utf-8")
    orphan_xml.write_text(_voc_xml("shared.jpg", [("car", (20, 20, 40, 40))]), encoding="utf-8")

    dialog = CleanupDialog([], default_source=str(root))
    try:
        dialog._start_scan()
        assert _wait(app, lambda: not dialog._scanning)
        assert dialog._to_delete_images == [empty_image]
        assert dialog._to_delete_annotations == [orphan_xml]
        with locked_file(empty_image):
            lines = run_cleanup(dialog, quality=True)

        assert any("未能删除" in line for line in lines), lines
        assert keep_image.exists() and keep_xml.exists()
        assert empty_image.exists()
        assert empty_xml.exists(), "paired XML must survive when image deletion fails"
        assert not orphan_xml.exists(), "independent orphan cleanup must still proceed"
    finally:
        dialog.close()


def test_coco_cleanup_mixed_delete_success_only_removes_successful_json_records(
    tmp_path, run_cleanup, locked_file,
):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "coco-mixed-delete"
    success_image = root / "images" / "train" / "empty.jpg"
    failed_image = root / "images" / "val" / "empty.jpg"
    keep_image = root / "images" / "val" / "keep.jpg"
    _image(success_image, (170, 40, 40))
    _image(failed_image, (40, 170, 40))
    _image(keep_image, (40, 40, 170))
    (root / "annotations").mkdir(parents=True)
    json_path = root / "annotations" / "annotations.json"
    document = {
        "images": [
            {"id": 1, "file_name": "train/empty.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "val/empty.jpg", "width": 100, "height": 80},
            {"id": 3, "file_name": "val/keep.jpg", "width": 100, "height": 80},
        ],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {"id": 9, "image_id": 3, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}
        ],
    }
    json_path.write_text(json.dumps(document), encoding="utf-8")

    dialog = CleanupDialog([], default_source=str(root))
    try:
        dialog._start_scan()
        assert _wait(app, lambda: not dialog._scanning)
        assert dialog._to_delete_images == [success_image, failed_image]
        with locked_file(failed_image):
            lines = run_cleanup(dialog, quality=True)

        assert any("未能删除" in line for line in lines), lines
        assert not success_image.exists()
        assert failed_image.exists()
        assert keep_image.exists()
        # Only the file that really went is reported as cleaned.
        cleaned_names = {
            line.removeprefix("[已清理] ").strip() for line in lines
            if line.startswith("[已清理]")
        }
        assert cleaned_names == {"empty.jpg"}, cleaned_names
        cleaned = json.loads(json_path.read_text(encoding="utf-8"))
        # The one record whose file survived keeps its record: the JSON still
        # describes the dataset that is on disk.
        assert [item["file_name"] for item in cleaned["images"]] == ["val/empty.jpg", "val/keep.jpg"]
        assert cleaned["annotations"] == document["annotations"]
    finally:
        dialog.close()


def test_annotation_context_path_never_uses_ambiguous_same_stem_file(tmp_path):
    from types import SimpleNamespace

    from src.models.project import ImageRecord
    from src.widgets.main_window import MainWindow

    image_root = tmp_path / "images"
    annotation_root = tmp_path / "labels"
    target_image = image_root / "missing" / "shared.jpg"
    _image(target_image)
    for folder in ("train", "val"):
        path = annotation_root / folder / "shared.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    record = ImageRecord(target_image, 100, 80, "JPEG", target_image.stat().st_size)
    owner = SimpleNamespace(settings=SimpleNamespace(
        annotation_dir=str(annotation_root),
        annotation_format="yolo",
        image_dir=str(image_root),
    ))

    assert MainWindow._annotation_path_for_record(owner, record) is None

    (annotation_root / "val" / "shared.txt").unlink()
    assert MainWindow._annotation_path_for_record(owner, record) == annotation_root / "train" / "shared.txt"

    exact = annotation_root / "missing" / "shared.txt"
    exact.parent.mkdir(parents=True)
    exact.write_text("", encoding="utf-8")
    assert MainWindow._annotation_path_for_record(owner, record) == exact
