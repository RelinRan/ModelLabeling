"""Generated data only: interoperability and repair safety for every YOLO task."""
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image
from PySide6.QtCore import QPointF

from src.models.annotation import Annotation, LabelPreset, ShapeType
from src.models.project import ImageRecord, ProjectSettings
from src.services.annotation_service import AnnotationService
from src.services.cleanup import scan_dataset
from src.services.conversion_service import ConversionOptions, ConversionService
from src.services.dataset_detector import DatasetDetector
from src.services.onnx_service import YoloOnnxDetector
from src.services.segmentation_geometry import bridge_contours
from src.services.video_tools import synthesize_dataset
from src.services.workers import AutoLabelWorker
from src.services.workers import DatasetStatisticsWorker
from src.services.dataset_initializer import DatasetInitializer
from src.services.yolo_rows import parse_yolo_row, YoloRowError


ROWS = {
    "yolo_detection": "0 0.5 0.5 0.4 0.4",
    "yolo_segmentation": "0 0 0 0 0.5 0.5 0.5",
    "yolo_pose": "0 0.5 0.5 0.4 0.4 0.5 0.5 2",
    "yolo_obb": "0 0 0 0 0.5 0.5 0.5 0.5 0",
}
TASK_NAMES = {"yolo_detection": "detect", "yolo_segmentation": "segment",
              "yolo_pose": "pose", "yolo_obb": "obb"}
PRESETS = [LabelPreset("object", 0, "#00e5ff")]


def dataset(root, task, dimensions=3):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    image = root / "images" / "frame.jpg"
    Image.new("RGB", (640, 480), "gray").save(image)
    metadata = {"task": TASK_NAMES[task], "names": ["object"], "train": "images", "val": "images"}
    row = ROWS[task]
    if task == "yolo_pose":
        metadata.update(kpt_shape=[1, dimensions], kpt_names=["center"], flip_idx=[0])
        if dimensions == 2:
            row = " ".join(row.split()[:-1])
    (root / "data.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    (root / "labels" / "frame.txt").write_text(row + "\n", encoding="utf-8")
    settings = ProjectSettings(image_dir=root / "images", annotation_dir=root / "labels",
                               annotation_format="yolo", dataset_task=task, label_presets=PRESETS)
    return image, settings


@pytest.mark.parametrize("task", ROWS)
def test_cleanup_preserves_valid_standard_rows(tmp_path, task):
    root = tmp_path / task
    dataset(root, task)
    issues = scan_dataset(root).issues
    assert not [issue.rule for issue in issues if issue.rule.startswith("yolo.row.")]


@pytest.mark.parametrize("task", ROWS)
def test_load_edit_save_convert_synthesize_reopen(tmp_path, task):
    source = tmp_path / "source"
    image, settings = dataset(source, task)
    service = AnnotationService()
    loaded = service.load(image, settings.annotation_dir, settings)
    assert not loaded.error
    assert len(loaded.annotations) == 1
    original = loaded.annotations[0]
    assert service.save(image, loaded.annotations, settings.annotation_dir, settings).ok
    for operation in ("convert", "synthesis"):
        output = tmp_path / operation
        if operation == "convert":
            report = ConversionService().convert(ConversionOptions("yolo", source, "yolo", output, []))
        else:
            report = synthesize_dataset(source, output, "yolo", [])
        assert report.succeeded == 1 and report.failed == 0, report.errors
        detected = DatasetDetector.detect(output)
        assert detected.task_name == task
        out_settings = ProjectSettings(image_dir=detected.image_dir, annotation_dir=detected.annotation_dir,
                                        annotation_format="yolo", dataset_task=task, label_presets=PRESETS)
        restored = service.load(detected.image_dir / "frame.jpg", detected.annotation_dir, out_settings)
        assert not restored.error
        annotation = restored.annotations[0]
        assert annotation.label == original.label
        assert annotation.shape_type == original.shape_type
        assert [(p.x(), p.y()) for p in annotation.points] == pytest.approx(
            [(p.x(), p.y()) for p in original.points])
        assert [k.visibility for k in annotation.keypoints] == [k.visibility for k in original.keypoints]


def test_pose_two_dimensions_remain_two_after_edit_and_conversion(tmp_path):
    image, settings = dataset(tmp_path / "source", "yolo_pose", 2)
    service = AnnotationService()
    loaded = service.load(image, settings.annotation_dir, settings)
    assert not loaded.error
    assert service.save(image, loaded.annotations, settings.annotation_dir, settings).ok
    assert len((settings.annotation_dir / "frame.txt").read_text().split()) == 7
    output = tmp_path / "output"
    report = ConversionService().convert(ConversionOptions("yolo", image.parent.parent, "yolo", output, []))
    assert report.failed == 0, report.errors
    metadata = yaml.safe_load((output / "data.yaml").read_text())
    assert metadata["kpt_shape"] == [1, 2]
    assert metadata["flip_idx"] == [0]
    assert len((output / "labels" / "frame.txt").read_text().split()) == 7
    loaded.annotations[0].keypoints[0].visibility = 0
    assert not service.save(image, loaded.annotations, settings.annotation_dir, settings).ok


def test_pose_edit_rejects_schema_reordering_without_changing_file(tmp_path):
    image, settings = dataset(tmp_path / "source", "yolo_pose")
    service = AnnotationService()
    loaded = service.load(image, settings.annotation_dir, settings)
    before = (settings.annotation_dir / "frame.txt").read_bytes()
    loaded.annotations[0].keypoints[0].name = "different_point"
    saved = service.save(image, loaded.annotations, settings.annotation_dir, settings)
    assert not saved.ok and "kpt_names" in saved.error
    assert (settings.annotation_dir / "frame.txt").read_bytes() == before


def test_yolo_metadata_writer_preserves_open_dataset_paths_only_when_requested(tmp_path):
    root = tmp_path / "source"
    dataset(root, "yolo_detection")
    path = root / "data.yaml"
    document = yaml.safe_load(path.read_text())
    document.update(train="images/train", val="images/val")
    path.write_text(yaml.safe_dump(document))
    ConversionService._write_yolo_yaml(root, PRESETS, "yolo_detection", preserve_paths=True)
    assert yaml.safe_load(path.read_text())["train"] == "images/train"
    ConversionService._write_yolo_yaml(root, PRESETS, "yolo_detection")
    assert yaml.safe_load(path.read_text())["train"] == "images"


@pytest.mark.parametrize("task", ROWS)
@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_nonfinite_rows_are_rejected(task, bad):
    row = ROWS[task].split()
    row[1] = bad
    with pytest.raises(YoloRowError, match="finite"):
        parse_yolo_row(" ".join(row), task)


def test_cleanup_does_not_merge_distinct_poses_with_same_box(tmp_path):
    root = tmp_path / "source"
    _, settings = dataset(root, "yolo_pose")
    path = settings.annotation_dir / "frame.txt"
    path.write_text(ROWS["yolo_pose"] + "\n0 0.5 0.5 0.4 0.4 0.6 0.6 2\n")
    assert not [i for i in scan_dataset(root).issues if "duplicate" in i.rule]


class Session:
    def __init__(self, output):
        self.output = output

    def get_inputs(self):
        return [SimpleNamespace(name="images", shape=[1, 3, 640, 640])]

    def run(self, names, values):
        return [self.output]


@pytest.mark.parametrize("end_to_end", [False, True])
def test_obb_onnx_rotated_nms_and_letterbox(end_to_end):
    raw = [320, 320, 200, 100, 0.9, np.pi / 4]
    e2e = [320, 320, 200, 100, 0.9, 0, np.pi / 4]
    output = np.array([[e2e, e2e]], np.float32) if end_to_end else np.array([[raw, raw]], np.float32).transpose(0, 2, 1)
    detector = YoloOnnxDetector(Session(output))
    detector.task = "obb"
    detector.class_names = ["object"]
    result = detector.predict(Image.new("RGB", (1280, 720)), PRESETS, letterbox=True)
    assert len(result) == 1
    annotation = result[0]
    assert annotation.shape_type == ShapeType.OBB
    points = np.array([(p.x(), p.y()) for p in annotation.points], np.float32)
    expected = cv2.boxPoints(((640, 360), (400, 200), 45))
    _, overlap = cv2.intersectConvexConvex(points, expected)
    assert overlap is not None
    assert abs(cv2.contourArea(points) - 80000) < 0.1
    assert cv2.intersectConvexConvex(points, expected)[0] / 80000 > 0.9999


def test_pose_onnx_kpt_shape_without_names_and_two_dimensions():
    detector = YoloOnnxDetector(Session(np.array([320, 320, 100, 100, 0.9, 300, 310], np.float32).reshape(1, 7, 1)))
    detector.task = "pose"
    detector.class_names = ["object"]
    detector.keypoint_shape = (1, 2)
    result = detector.predict(Image.new("RGB", (640, 640)), PRESETS)
    assert len(result) == 1
    assert result[0].keypoints[0].point == QPointF(300, 310)
    assert result[0].keypoints[0].visibility == 2


def test_single_class_raw_detection_onnx():
    detector = YoloOnnxDetector(Session(np.array([320, 320, 100, 100, 0.9], np.float32).reshape(1, 5, 1)))
    result = detector.predict(Image.new("RGB", (640, 640)), PRESETS)
    assert len(result) == 1
    assert result[0].points == [QPointF(270, 270), QPointF(370, 370)]


def test_disconnected_contours_keep_pixel_mask_and_save(tmp_path):
    parts = [[QPointF(10, 10), QPointF(60, 10), QPointF(60, 60), QPointF(10, 60)],
             [QPointF(100, 100), QPointF(150, 100), QPointF(150, 150), QPointF(100, 150)]]
    points = bridge_contours(parts)
    expected, actual = np.zeros((200, 200), np.uint8), np.zeros((200, 200), np.uint8)
    contours = [np.array([(p.x(), p.y()) for p in part], np.int32) for part in parts]
    cv2.fillPoly(expected, contours, 1)
    cv2.fillPoly(actual, [np.array([(p.x(), p.y()) for p in points], np.int32)], 1)
    intersection = np.logical_and(expected, actual).sum()
    union = np.logical_or(expected, actual).sum()
    assert intersection / union > 0.99
    image, settings = dataset(tmp_path / "source", "yolo_segmentation")
    annotation = Annotation(ShapeType.POLYGON, "object", points)
    assert AnnotationService().save(image, [annotation], settings.annotation_dir, settings).ok


@pytest.mark.parametrize("task", ROWS)
def test_auto_label_worker_writes_reopenable_task_metadata(tmp_path, monkeypatch, task):
    image, settings = dataset(tmp_path / "source", task)
    loaded = AnnotationService().load(image, settings.annotation_dir, settings)
    class Detector:
        class_names = ["object"]
        keypoint_names = ["center"] if task == "yolo_pose" else []
        keypoint_shape = (1, 3) if task == "yolo_pose" else None

        def load(self, path):
            pass

        def predict(self, *args, **kwargs):
            assert kwargs["letterbox"]
            return loaded.annotations

    Detector.task = TASK_NAMES[task]
    monkeypatch.setattr("src.services.workers.YoloOnnxDetector", Detector)
    settings.onnx_model_path = tmp_path / "fake.onnx"
    record = ImageRecord(image, 640, 480, "JPEG", image.stat().st_size)
    worker = AutoLabelWorker([record], settings)
    errors, progress = [], []
    worker.failed.connect(errors.append)
    worker.progress.connect(lambda a, b: progress.append((a, b)))
    worker.run()
    assert not errors
    assert progress == [(1, 1)]
    assert DatasetDetector.detect(image.parent.parent).task_name == task


def test_obb_conversion_to_standard_polygon_preserves_corners(tmp_path):
    image, settings = dataset(tmp_path / "source", "yolo_obb")
    output = tmp_path / "segment"
    report = ConversionService().convert(ConversionOptions("yolo", image.parent.parent, "yolo", output, [],
                                                           output_task="yolo_segmentation"))
    assert report.failed == 0, report.errors
    assert list(map(float, (output / "labels" / "frame.txt").read_text().split())) == list(map(float, ROWS["yolo_obb"].split()))
    assert DatasetDetector.detect(output).task_name == "yolo_segmentation"


@pytest.mark.parametrize("task", ROWS)
def test_workspace_import_retains_standard_task_and_schema(tmp_path, task):
    image, settings = dataset(tmp_path / "source", task)
    info = DatasetInitializer.create_in_workspace(tmp_path / "workspace", "imported", task,
                                                  source_dir=image.parent.parent)
    assert info.copied_images == 1 and info.imported_annotations == 1
    assert DatasetDetector.detect(info.root).task_name == task
    if task == "yolo_pose":
        assert yaml.safe_load((info.root / "data.yaml").read_text())["kpt_shape"] == [1, 3]


def test_multipart_conversion_requires_explicit_option_and_keeps_parts(tmp_path):
    image, settings = dataset(tmp_path / "source", "yolo_segmentation")
    parts = [[QPointF(10, 10), QPointF(60, 10), QPointF(60, 60), QPointF(10, 60)],
             [QPointF(100, 100), QPointF(150, 100), QPointF(150, 150), QPointF(100, 150)]]
    annotation = Annotation(ShapeType.POLYGON, "object", parts[0], polygon_parts=parts)
    coco = tmp_path / "coco"
    (coco / "images").mkdir(parents=True)
    with Image.open(image) as source:
        source.save(coco / "images" / "frame.jpg")
    AnnotationService().save_coco_batch([(coco / "images" / "frame.jpg", [annotation])], coco / "annotations", PRESETS)
    rejected = ConversionService().convert(ConversionOptions("coco", coco, "yolo", tmp_path / "rejected", [],
                                                             output_task="yolo_segmentation"))
    assert rejected.failed == 1
    output = tmp_path / "merged"
    accepted = ConversionService().convert(ConversionOptions("coco", coco, "yolo", output, [],
                                                             output_task="yolo_segmentation", merge_polygon_parts=True))
    assert accepted.failed == 0 and accepted.succeeded == 1, accepted.errors
    settings.image_dir, settings.annotation_dir = output / "images", output / "labels"
    restored = AnnotationService().load(output / "images" / "frame.jpg", output / "labels", settings)
    assert not restored.error
    assert len(restored.annotations) == 1
    assert len(restored.annotations[0].points) >= 10


def test_pose_model_metadata_shape_generates_missing_point_names(monkeypatch):
    import sys
    session = Session(np.array([320, 320, 100, 100, 0.9, 300, 310], np.float32).reshape(1, 7, 1))
    session.get_modelmeta = lambda: SimpleNamespace(custom_metadata_map={"task": "pose", "names": "{0: 'object'}", "kpt_shape": "[1, 2]"})
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=lambda *a, **k: session))
    detector = YoloOnnxDetector()
    detector.load(Path("fake.onnx"))
    assert detector.keypoint_names == ["keypoint_0"]
    assert detector.keypoint_shape == (1, 2)
    assert len(detector.predict(Image.new("RGB", (640, 640)), PRESETS)) == 1


def test_conversion_and_synthesis_dialog_offer_all_yolo_tasks():
    from PySide6.QtWidgets import QApplication
    from src.widgets.conversion_dialog import ConversionDialog
    from src.widgets.video_tools_dialogs import DatasetSynthesisDialog
    app = QApplication.instance() or QApplication([])
    dialog = ConversionDialog(PRESETS)
    dialog.output_format.setCurrentIndex(dialog.output_format.findData("yolo"))
    assert {dialog.output_task.itemData(i) for i in range(dialog.output_task.count())} == set(ROWS)
    dialog.output_task.setCurrentIndex(dialog.output_task.findData("yolo_segmentation"))
    assert not dialog.merge_parts.isHidden()
    dialog.close()
    synthesis = DatasetSynthesisDialog()
    synthesis.format.setCurrentIndex(synthesis.format.findData("yolo"))
    # Every YOLO task is offered explicitly; there is no "follow source"
    # sentinel, so the user always chooses the output task.
    assert {synthesis.task.itemData(i) for i in range(synthesis.task.count())} == set(ROWS)
    synthesis.close()


@pytest.mark.parametrize("task", ROWS)
def test_complete_dataset_statistics_for_all_tasks(tmp_path, task):
    image, settings = dataset(tmp_path / "source", task)
    record = ImageRecord(image, 640, 480, "JPEG", image.stat().st_size)
    worker = DatasetStatisticsWorker([record], settings.image_dir, settings.annotation_dir, settings, PRESETS)
    errors, results = [], []
    worker.failed.connect(errors.append)
    worker.finished.connect(results.append)
    worker.run()
    assert not errors
    assert results[0]["total_images"] == 1
    assert results[0]["labeled_images"] == 1
    assert results[0]["total_labels"] == 1
    assert results[0]["label_counts"] == {"object": 1}


def test_gui_open_switch_relabel_autosave_reopen_all_yolo_tasks(tmp_path, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from src.widgets.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr("src.widgets.main_window.AppDialog.information", classmethod(lambda *a, **k: None))
    monkeypatch.setattr("src.widgets.main_window.AppDialog.question", classmethod(lambda *a, **k: True))

    def wait_for(predicate):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError("GUI workflow timed out")

    window = MainWindow()
    try:
        for task in ROWS:
            root = tmp_path / task
            image, settings = dataset(root, task)
            window._start_open_path(root)
            wait_for(lambda: window._dataset_scan_completed and window._dataset_thread is None)
            wait_for(lambda: window.state.current_image is not None and len(window.canvas.annotations) == 1)
            assert window.state.current_image.path == image
            assert window.settings.dataset_task == task
            window.canvas.setFocus()
            QTest.keyClick(window.canvas, Qt.Key.Key_W)
            assert window.canvas.draw_enabled
            QTest.keyClick(window.canvas, Qt.Key.Key_W)
            assert not window.canvas.draw_enabled
            window.canvas.annotation_items[0].setSelected(True)
            assert window.canvas.update_selected_label("changed")
            wait_for(lambda: not window.dirty and (window._save_thread is None or not window._save_thread.is_alive()))
            assert window.state.current_image.annotations[0].label == "changed"
            saved_settings = ProjectSettings.from_dict(window.settings.to_dict())
            saved_settings.label_presets = list(window.dataset_parser_presets or window.settings.label_presets)
            parsed = AnnotationService().load(image, settings.annotation_dir, saved_settings)
            assert not parsed.error
            assert parsed.annotations[0].label == "changed"
        for task in ROWS:
            window._start_open_path(tmp_path / task)
            wait_for(lambda: window._dataset_scan_completed and window._dataset_thread is None)
            wait_for(lambda: len(window.canvas.annotations) == 1)
            assert window.canvas.annotations[0].label == "changed"
            assert window.settings.dataset_task == task
    finally:
        window.close()


def test_conversion_cancel_keeps_existing_target_untouched(tmp_path):
    source = tmp_path / "source"
    dataset(source, "yolo_detection")
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("original", encoding="utf-8")

    report = ConversionService().convert(
        ConversionOptions("yolo", source, "yolo", output, [], overwrite=True),
        cancel_callback=lambda: True,
    )

    assert report.cancelled
    assert report.failed == 0
    assert marker.read_text(encoding="utf-8") == "original"
    assert not (output / "data.yaml").exists()
    assert not list(output.parent.glob(f".{output.name}.staging-*"))


def test_conversion_publishes_final_dataset_path_when_merging_existing_target(tmp_path):
    source = tmp_path / "source"
    dataset(source, "yolo_detection")
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("original", encoding="utf-8")

    report = ConversionService().convert(
        ConversionOptions("yolo", source, "yolo", output, [], overwrite=False),
    )

    assert report.failed == 0, report.errors
    assert (output / "keep.txt").read_text(encoding="utf-8") == "original"
    metadata = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert Path(metadata["path"]).resolve() == output.resolve()
    assert not list(output.parent.glob(f".{output.name}.staging-*"))


@pytest.mark.parametrize("output_format", ["yolo", "voc", "coco"])
def test_cancel_after_processing_keeps_target_unchanged(tmp_path, output_format):
    source = tmp_path / "source"
    dataset(source, "yolo_detection")
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.bin").write_bytes(b"original")
    cancelled = False

    def progress(current, total):
        nonlocal cancelled
        cancelled = current > 0

    report = ConversionService().convert(
        ConversionOptions("yolo", source, output_format, output, [], overwrite=True),
        progress_callback=progress, cancel_callback=lambda: cancelled,
    )
    assert report.cancelled
    assert report.succeeded == 1
    assert list(output.iterdir()) == [output / "keep.bin"]
    assert (output / "keep.bin").read_bytes() == b"original"
    assert not list(tmp_path.glob(".output.staging-*"))


def test_conversion_overwrite_preserves_unrelated_files(tmp_path):
    source = tmp_path / "source"
    dataset(source, "yolo_detection")
    output = tmp_path / "output"
    (output / "labels" / "nested").mkdir(parents=True)
    (output / "labels" / "nested" / "stale.txt").write_text("stale")
    (output / "notes.md").write_text("keep")
    report = ConversionService().convert(
        ConversionOptions("yolo", source, "yolo", output, [], overwrite=True))
    assert not report.failed, report.errors
    assert not (output / "labels" / "nested" / "stale.txt").exists()
    assert (output / "notes.md").read_text() == "keep"
    assert (output / "labels" / "frame.txt").is_file()


def test_conversion_publish_failure_restores_old_target(tmp_path, monkeypatch):
    source = tmp_path / "source"
    dataset(source, "yolo_detection")
    output = tmp_path / "output"
    output.mkdir()
    (output / "old.txt").write_text("original")
    original_replace = Path.replace

    def fail_publish(path, target):
        if path.name == "dataset" and Path(target) == output:
            raise OSError("simulated publish failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publish)
    report = ConversionService().convert(
        ConversionOptions("yolo", source, "yolo", output, [], overwrite=True))
    assert report.failed
    assert "simulated publish failure" in report.errors[0]
    assert (output / "old.txt").read_text() == "original"
    assert not (output / "data.yaml").exists()
    assert not list(tmp_path.glob(".output.*"))


def test_conversion_unexpected_exception_cleans_stage(tmp_path):
    source = tmp_path / "source"
    dataset(source, "yolo_detection")

    def fail_progress(*args):
        raise RuntimeError("callback failure")

    with pytest.raises(RuntimeError, match="callback failure"):
        ConversionService().convert(
            ConversionOptions("yolo", source, "yolo", tmp_path / "output", []),
            progress_callback=fail_progress)
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.staging-*"))


def test_conversion_failure_does_not_publish_good_files_either(tmp_path):
    source = tmp_path / "source"
    image, settings = dataset(source, "yolo_detection")
    Image.new("RGB", (640, 480)).save(image.with_name("broken.jpg"))
    (settings.annotation_dir / "broken.txt").write_text("invalid row")
    output = tmp_path / "output"
    output.mkdir()
    (output / "old.txt").write_text("original")
    report = ConversionService().convert(
        ConversionOptions("yolo", source, "voc", output, [], overwrite=True))
    assert report.failed
    assert list(output.iterdir()) == [output / "old.txt"]
    assert not list(tmp_path.glob(".output.staging-*"))


def test_conversion_skip_keeps_pose_names_and_new_metadata(tmp_path):
    source = tmp_path / "source"
    dataset(source, "yolo_pose")
    output = tmp_path / "output"
    options = ConversionOptions("yolo", source, "yolo", output, [])
    assert not ConversionService().convert(options).failed
    report = ConversionService().convert(options)
    assert report.skipped == 1 and not report.failed
    document = yaml.safe_load((output / "data.yaml").read_text())
    assert document["kpt_names"] == ["center"]
    assert document["path"] == output.resolve().as_posix()


def test_conversion_rejects_flattened_filename_collision(tmp_path):
    source = tmp_path / "source"
    image, settings = dataset(source, "yolo_detection")
    nested = image.parent / "folder"
    nested.mkdir()
    Image.new("RGB", (640, 480)).save(nested / "frame.jpg")
    Image.new("RGB", (640, 480)).save(image.parent / "folder__frame.jpg")
    with pytest.raises(ValueError, match="same output file"):
        ConversionService().convert(ConversionOptions(
            "yolo", source, "yolo", tmp_path / "output", [], flatten_output=True))
    assert not (tmp_path / "output").exists()


def test_conversion_rejects_incompatible_existing_yolo_mapping(tmp_path):
    source = tmp_path / "source"
    dataset(source, "yolo_detection")
    output = tmp_path / "output"
    dataset(output, "yolo_detection")
    metadata = output / "data.yaml"
    document = yaml.safe_load(metadata.read_text())
    document["names"] = ["different"]
    metadata.write_text(yaml.safe_dump(document))
    before = metadata.read_bytes()
    with pytest.raises(ValueError, match="classes differ"):
        ConversionService().convert(ConversionOptions("yolo", source, "yolo", output, []))
    assert metadata.read_bytes() == before
    assert not list(tmp_path.glob(".output.staging-*"))
