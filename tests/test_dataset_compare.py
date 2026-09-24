"""The dataset/model comparison: ground truth, the report, and read-only-ness.

Every fixture builds its own dataset. The comparison writes a report beside the
dataset it reads, so a shared dataset would be a shared output file -- and the
point of several tests below is that nothing *else* in the dataset is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import QPointF

from src.models.annotation import Annotation, ShapeType
from src.services.dataset_compare import (
    NO_LABEL,
    CompareCase,
    GroundTruth,
    _model_presets,
    _predict_labels,
    compare_dataset,
    render_report,
    report_path_for,
)
from src.services.dataset_detector import DatasetDetector


def _image(path: Path, color=(60, 100, 140), size=(120, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")


def _voc_xml(path: Path, labels: list[str], filename: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    objects = "".join(
        f"<object><name>{label}</name><bndbox>"
        "<xmin>10</xmin><ymin>10</ymin><xmax>60</xmax><ymax>50</ymax>"
        "</bndbox></object>"
        for label in labels
    )
    path.write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n"
        f"<annotation><filename>{filename or path.stem + '.jpg'}</filename>"
        "<size><width>120</width><height>80</height><depth>3</depth></size>"
        f"{objects}</annotation>",
        encoding="utf-8",
    )


def _voc(root: Path, frames: list[tuple[str, str | None]], folder: str = "") -> Path:
    """A VOC dataset. ``None`` means an image whose annotation declares nothing."""
    for name, label in frames:
        _image(root / "JPEGImages" / folder / f"{name}.jpg")
        _voc_xml(root / "Annotations" / folder / f"{name}.xml", [] if label is None else [label])
    return root


def _yolo(root: Path, frames: list[tuple[str, int | None]], names: list[str]) -> Path:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for name, class_id in frames:
        _image(root / "images" / f"{name}.jpg")
        body = "" if class_id is None else f"{class_id} 0.5 0.5 0.4 0.4\n"
        (root / "labels" / f"{name}.txt").write_text(body, encoding="utf-8")
    (root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return root


def _coco(root: Path, frames: list[tuple[str, str | None]], categories=("cat", "dog")) -> Path:
    images, annotations = [], []
    for index, (name, label) in enumerate(frames, start=1):
        _image(root / "images" / f"{name}.jpg")
        images.append({"id": index, "file_name": f"{name}.jpg", "width": 120, "height": 80})
        if label is not None:
            annotations.append({
                "id": index, "image_id": index,
                "category_id": categories.index(label) + 1,
                "bbox": [10, 10, 50, 40], "area": 2000, "iscrowd": 0,
            })
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / "annotations.json").write_text(
        json.dumps({
            "images": images,
            "categories": [{"id": i, "name": n} for i, n in enumerate(categories, start=1)],
            "annotations": annotations,
        }),
        encoding="utf-8",
    )
    return root


def _answers(monkeypatch, answers: dict, names: list[str], root: Path) -> Path:
    """Point the comparison at a model whose output is known.

    Two seams, one per unit under test: ``YoloOnnxDetector`` supplies the class
    list the model declares, and ``_predict_labels`` supplies what it found --
    which is what lets a test state its expectations as ``{"a.jpg": [("3X", .8)]}``
    instead of interrogating a real export whose answers depend on what a network
    happened to learn.
    """
    model_path = root / "model.onnx"
    model_path.write_bytes(b"stub")

    class StubDetector:
        class_names = list(names)

        def load(self, path: Path) -> None:
            assert Path(path) == model_path

    def fake_predict(detector, presets, image_path, threshold, input_size, nms_threshold):
        found = answers[image_path.name]
        if isinstance(found, Exception):
            raise found
        return [(label, score) for label, score in found if score >= threshold]

    monkeypatch.setattr("src.services.dataset_compare.YoloOnnxDetector", StubDetector)
    monkeypatch.setattr("src.services.dataset_compare._predict_labels", fake_predict)
    return model_path


# ---------------------------- ground truth ----------------------------


def test_voc_ground_truth_reads_the_names_the_annotation_declares(tmp_path):
    root = _voc(tmp_path / "voc", [("a", "1X"), ("b", None), ("c", "3X")])
    truth = GroundTruth(DatasetDetector.detect(root))

    assert truth.labels(root / "JPEGImages" / "a.jpg") == ["1X"]
    assert truth.labels(root / "JPEGImages" / "b.jpg") == []
    assert truth.labels(root / "JPEGImages" / "c.jpg") == ["3X"]


def test_reading_coco_ground_truth_leaves_no_cache_in_the_dataset(tmp_path):
    """The comparison is a read. The app's loader would write a coco.db here."""
    root = _coco(tmp_path / "coco", [("a", "cat"), ("b", "dog")])
    before = {path.name for path in (root / "annotations").iterdir()}

    truth = GroundTruth(DatasetDetector.detect(root))

    assert truth.labels(root / "images" / "a.jpg") == ["cat"]
    assert truth.labels(root / "images" / "b.jpg") == ["dog"]
    assert {path.name for path in (root / "annotations").iterdir()} == before


def test_yolo_ground_truth_uses_the_dataset_class_table_not_the_model(tmp_path):
    """A class id only becomes a name through the dataset's own table.

    Reading it through the model's names would erase the very disagreement worth
    reporting: a dataset whose class list and model disagree.
    """
    root = _yolo(tmp_path / "yolo", [("a", 1), ("b", 0)], ["left", "right"])
    truth = GroundTruth(DatasetDetector.detect(root))

    assert truth.labels(root / "images" / "a.jpg") == ["right"]
    assert truth.labels(root / "images" / "b.jpg") == ["left"]


def test_a_nested_annotation_is_found_by_its_own_path(tmp_path):
    root = tmp_path / "nested"
    for folder in ("train", "val"):
        _voc(root, [("shared", "1X")], folder)

    truth = GroundTruth(DatasetDetector.detect(root))

    assert truth.labels(root / "JPEGImages" / "train" / "shared.jpg") == ["1X"]
    assert truth.labels(root / "JPEGImages" / "val" / "shared.jpg") == ["1X"]


def test_an_ambiguous_stem_is_not_paired_by_guess(tmp_path):
    """Two annotations claim the name ``shared``, so a bare-stem match is a guess.

    A wrong pairing would report a disagreement that is really a lookup accident,
    so the image reads as unlabelled instead.
    """
    root = tmp_path / "ambiguous"
    for folder in ("train", "val"):
        _voc(root, [("shared", "1X")], folder)
    _image(root / "JPEGImages" / "other" / "shared.jpg")

    truth = GroundTruth(DatasetDetector.detect(root))

    # Mirrored paths still resolve exactly ...
    assert truth.labels(root / "JPEGImages" / "train" / "shared.jpg") == ["1X"]
    # ... and the third image, which matches neither, falls back to nothing.
    assert truth.labels(root / "JPEGImages" / "other" / "shared.jpg") == []


def test_a_unique_stem_still_matches_across_directories(tmp_path):
    """The ambiguity rule above must not stop a unique name from resolving."""
    root = tmp_path / "flat"
    _voc(root, [("only", "3X")], "train")
    _image(root / "JPEGImages" / "val" / "only.jpg")

    truth = GroundTruth(DatasetDetector.detect(root))

    assert truth.labels(root / "JPEGImages" / "val" / "only.jpg") == ["3X"]


# ----------------------------- predictions -----------------------------


class _FixedDetector:
    """A detector that reports exactly these boxes, whatever it is shown."""

    def __init__(self, found: list[tuple[str, float]]) -> None:
        self.found = found

    def predict(self, image, presets, input_size=640, confidence_threshold=0.25,
                nms_threshold=0.45, letterbox=False):
        assert letterbox, "the comparison must feed the model the way it was trained"
        return [
            Annotation(ShapeType.RECTANGLE, label, [QPointF(10, 10), QPointF(60, 50)],
                       confidence=score, source="onnx")
            for label, score in self.found
        ]


def test_predict_labels_keeps_the_best_score_per_label(tmp_path):
    image = tmp_path / "a.jpg"
    _image(image)
    detector = _FixedDetector([("1X", 0.4), ("3X", 0.8), ("1X", 0.9), ("3X", 0.7)])

    found = _predict_labels(detector, [], image, 0.5, 640, 0.45)

    # One entry per label, the highest score, and best first.
    assert found == [("1X", 0.9), ("3X", 0.8)]


def test_predict_labels_orders_ties_by_name(tmp_path):
    image = tmp_path / "a.jpg"
    _image(image)
    detector = _FixedDetector([("3X", 0.8), ("1X", 0.8)])

    assert _predict_labels(detector, [], image, 0.5, 640, 0.45) == [("1X", 0.8), ("3X", 0.8)]


def test_presets_come_from_the_model_and_a_nameless_model_is_refused(tmp_path):
    class Named:
        class_names = ["1X", "3X"]

    class Nameless:
        class_names = []

    assert [preset.name for preset in _model_presets(Named())] == ["1X", "3X"]
    assert [preset.class_id for preset in _model_presets(Named())] == [0, 1]
    with pytest.raises(ValueError, match="names"):
        _model_presets(Nameless())


# ------------------------------- report -------------------------------


def test_a_disagreement_is_written_in_the_documented_form(tmp_path, monkeypatch):
    root = _voc(tmp_path / "voc", [("a", "1X"), ("b", "1X")])
    model_path = _answers(monkeypatch, {"a.jpg": [("3X", 0.7521)], "b.jpg": [("1X", 0.9)]},
                          ["1X", "3X"], root)

    report = compare_dataset(model_path, root, 0.5)

    assert (report.total, report.matched, report.mismatched, report.skipped) == (2, 1, 1, 0)
    assert report.output_path == tmp_path / "voc_compare.txt"
    assert report.output_path == report_path_for(root)
    text = report.output_path.read_text(encoding="utf-8-sig")
    assert "a.jpg 1X -> 3X 0.75" in text, text
    assert "b.jpg" not in text, "a matching image is not a report line"


def test_the_report_beside_the_dataset_is_the_only_file_written(tmp_path, monkeypatch):
    root = _voc(tmp_path / "voc", [("a", "1X")])
    model_path = _answers(monkeypatch, {"a.jpg": [("3X", 0.8)]}, ["1X", "3X"], root)
    before = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    original = (root / "Annotations" / "a.xml").read_bytes()

    compare_dataset(model_path, root, 0.5)

    after = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    assert after == before, "the dataset itself must come back unchanged"
    assert (root / "Annotations" / "a.xml").read_bytes() == original


def test_every_line_that_is_not_a_mismatch_starts_with_a_hash(tmp_path, monkeypatch):
    """The file's whole contract: drop the ``#`` lines and the rest is data."""
    root = _voc(tmp_path / "voc", [("a", "1X"), ("b", None), ("c", "1X")])
    model_path = _answers(
        monkeypatch,
        {"a.jpg": [], "b.jpg": [("3X", 0.6)], "c.jpg": [("1X", 0.7)]},
        ["1X", "3X"], root,
    )

    report = compare_dataset(model_path, root, 0.5)

    lines = report.output_path.read_text(encoding="utf-8-sig").splitlines()
    data = [line for line in lines if not line.startswith("#")]
    # a.jpg: the model found nothing. b.jpg: nothing was labelled, the model did.
    assert data == ["a.jpg 1X -> None", "b.jpg None -> 3X 0.60"], data
    header = [line for line in lines if line.startswith("#")]
    assert any("总图片：3" in line for line in header)
    assert any("不一致：2" in line for line in header)


def test_the_threshold_decides_what_counts_as_detected(tmp_path, monkeypatch):
    root = _voc(tmp_path / "voc", [("a", "3X")])
    model_path = _answers(monkeypatch, {"a.jpg": [("3X", 0.94), ("1X", 0.7)]}, ["1X", "3X"], root)

    lenient = compare_dataset(model_path, root, 0.5)
    assert (lenient.matched, lenient.mismatched) == (0, 1)

    strict = compare_dataset(model_path, root, 0.9)
    assert (strict.matched, strict.mismatched) == (1, 0)


def test_a_cancelled_run_writes_no_report(tmp_path, monkeypatch):
    """A partial report reads exactly like a clean one; only one is worth keeping."""
    root = _voc(tmp_path / "voc", [("a", "1X"), ("b", "1X")])
    model_path = _answers(monkeypatch, {"a.jpg": [("3X", 0.8)], "b.jpg": [("3X", 0.8)]},
                          ["1X", "3X"], root)

    report = compare_dataset(model_path, root, 0.5, cancel_callback=lambda: True)

    assert report.cancelled
    assert not report.output_path.exists()
    assert report.total == 2 and report.compared == 0


def test_an_unreadable_image_is_skipped_not_fatal(tmp_path, monkeypatch):
    root = _voc(tmp_path / "voc", [("a", "1X"), ("b", "1X")])
    model_path = _answers(monkeypatch, {"a.jpg": [("1X", 0.9)], "b.jpg": OSError("broken image")},
                          ["1X", "3X"], root)

    report = compare_dataset(model_path, root, 0.5)

    assert (report.matched, report.skipped) == (1, 1)
    # What could not be read is not quietly counted as agreement.
    assert report.total == report.matched + report.mismatched + report.skipped
    assert any("broken image" in error for error in report.errors)
    assert "读取失败：1" in report.output_path.read_text(encoding="utf-8-sig")


def test_a_model_without_class_names_is_refused(tmp_path, monkeypatch):
    """Otherwise every image comes back unlabelled and nothing is wrong with the
    dataset -- an answer that looks like a clean run."""
    root = _voc(tmp_path / "voc", [("a", "1X")])
    model_path = _answers(monkeypatch, {}, [], root)

    with pytest.raises(ValueError, match="names"):
        compare_dataset(model_path, root, 0.5)
    assert not (tmp_path / "voc_compare.txt").exists()


def test_a_missing_model_or_dataset_says_which_one(tmp_path):
    root = _voc(tmp_path / "voc", [("a", "1X")])
    with pytest.raises(ValueError, match="模型文件不存在"):
        compare_dataset(tmp_path / "absent.onnx", root, 0.5)
    (tmp_path / "model.onnx").write_bytes(b"stub")
    with pytest.raises(ValueError, match="数据集目录不存在"):
        compare_dataset(tmp_path / "model.onnx", tmp_path / "absent", 0.5)


def test_per_class_counts_images_not_labels(tmp_path, monkeypatch):
    root = _voc(tmp_path / "voc", [("a", "1X"), ("b", "1X"), ("c", "3X")])
    model_path = _answers(
        monkeypatch,
        {"a.jpg": [("1X", 0.9)], "b.jpg": [("3X", 0.8)], "c.jpg": [("3X", 0.8)]},
        ["1X", "3X"], root,
    )

    report = compare_dataset(model_path, root, 0.5)

    assert report.per_class["1X"].images == 2
    assert report.per_class["1X"].mismatched == 1
    assert report.per_class["1X"].rate == 0.5
    assert report.per_class["3X"].mismatched == 0


def test_a_model_that_detects_nothing_is_a_mismatch_not_an_error(tmp_path, monkeypatch):
    root = _voc(tmp_path / "voc", [("a", "1X")])
    model_path = _answers(monkeypatch, {"a.jpg": []}, ["1X", "3X"], root)

    report = compare_dataset(model_path, root, 0.5)

    assert report.mismatched == 1
    assert report.errors == []
    assert "a.jpg 1X -> None" in report.output_path.read_text(encoding="utf-8-sig")


def test_the_case_line_orders_labels_by_score():
    case = CompareCase("a.jpg", ("1X", "3X"), ("3X", "1X"), (0.9, 0.4))
    assert case.line() == "a.jpg 1X+3X -> 3X+1X 0.90+0.40"
    assert CompareCase("a.jpg", (), (), ()).line() == f"a.jpg {NO_LABEL} -> {NO_LABEL}"


def test_the_same_label_twice_is_one_label():
    """Two boxes of a class are one class as far as "do the labels agree" goes."""
    assert CompareCase("a.jpg", ("1X",), ("1X", "1X"), (0.9, 0.8)).matched


def test_a_truncated_error_list_still_reports_the_full_count(tmp_path, monkeypatch):
    from src.services import dataset_compare

    root = _voc(tmp_path / "voc", [(f"f{index:03d}", "1X") for index in range(60)])
    model_path = _answers(
        monkeypatch,
        {f"f{index:03d}.jpg": OSError("nope") for index in range(60)},
        ["1X"], root,
    )

    report = compare_dataset(model_path, root, 0.5)

    assert report.skipped == 60
    assert len(report.errors) == dataset_compare.MAX_ERRORS == 50
    text = report.output_path.read_text(encoding="utf-8-sig")
    assert "读取失败：60" in text
    assert "另有 10 个未列出" in text


def test_the_renderer_says_nothing_about_classes_that_all_agree(tmp_path, monkeypatch):
    root = _voc(tmp_path / "voc", [("a", "1X")])
    model_path = _answers(monkeypatch, {"a.jpg": [("1X", 0.9)]}, ["1X", "3X"], root)
    report = compare_dataset(model_path, root, 0.5)

    text = render_report(report, [])
    assert "各类不一致" not in text
    assert "不一致：0" in text
    assert "不一致率：0.00%" in text
