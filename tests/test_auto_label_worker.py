from pathlib import Path

from src.models.annotation import LabelPreset
from src.models.project import ProjectSettings
from src.services.workers import AutoLabelWorker
from src.services.dataset_index import DatasetIndexRepository
from src.services.dataset_index import IndexedImage


def test_auto_label_model_class_names_are_authoritative():
    settings = ProjectSettings(
        label_presets=[
            LabelPreset("wrong-zero", 0, "#111111"),
            LabelPreset("wrong-one", 1, "#222222"),
        ],
    )
    worker = AutoLabelWorker([], settings)

    presets = worker._model_presets(["person", "vehicle", "NONE"])

    assert [preset.name for preset in presets] == ["person", "vehicle", "NONE"]
    assert [preset.class_id for preset in presets] == [0, 1, 2]
    assert presets[0].color == "#111111"


def test_pose_task_only_switches_yolo_dataset(monkeypatch, tmp_path):
    class Detector:
        task = "pose"
        keypoint_names = ["nose"]
        class_names = ["person"]

        def load(self, path):
            return None

    monkeypatch.setattr("src.services.workers.YoloOnnxDetector", Detector)
    for annotation_format, expected_task in (("yolo", "yolo_pose"), ("coco", "coco"), ("voc", "voc")):
        settings = ProjectSettings(
            annotation_format=annotation_format,
            dataset_task=expected_task if annotation_format != "yolo" else "yolo_detection",
            onnx_model_path=Path(tmp_path / "model.onnx"),
            label_presets=[LabelPreset("person", 0, "#00e5ff")],
        )
        worker = AutoLabelWorker([], settings)
        worker.run()
        assert worker.settings.dataset_task == expected_task


def test_segmentation_task_only_switches_yolo_dataset(monkeypatch, tmp_path):
    class Detector:
        task = "segment"
        keypoint_names = []
        class_names = ["person"]

        def load(self, path):
            return None

    monkeypatch.setattr("src.services.workers.YoloOnnxDetector", Detector)
    cases = (
        ("yolo", "yolo_segmentation", False),
        ("coco", "coco", False),
        ("voc", "voc", True),
    )
    for annotation_format, expected_task, should_fail in cases:
        settings = ProjectSettings(
            annotation_format=annotation_format,
            dataset_task=expected_task if annotation_format != "yolo" else "yolo_detection",
            onnx_model_path=Path(tmp_path / "model.onnx"),
            label_presets=[LabelPreset("person", 0, "#00e5ff")],
        )
        worker = AutoLabelWorker([], settings)
        errors = []
        worker.failed.connect(errors.append)
        worker.run()
        assert worker.settings.dataset_task == expected_task
        assert bool(errors) is should_fail


def test_auto_label_rejects_mismatched_yolo_class_schema(monkeypatch, tmp_path):
    class Detector:
        task = "detect"
        keypoint_names = []
        class_names = ["model-person", "model-vehicle"]

        def load(self, path):
            return None

    monkeypatch.setattr("src.services.workers.YoloOnnxDetector", Detector)
    settings = ProjectSettings(
        annotation_format="yolo", dataset_task="yolo_detection",
        onnx_model_path=Path(tmp_path / "model.onnx"),
        label_presets=[LabelPreset("dataset-person", 0, "#00e5ff")],
    )
    worker = AutoLabelWorker([], settings)
    errors = []
    worker.failed.connect(errors.append)

    worker.run()

    assert errors and "do not match" in errors[0]


def test_auto_label_reads_complete_dataset_from_index(tmp_path):
    image_dir = tmp_path / "images"
    annotation_dir = tmp_path / "labels"
    image_dir.mkdir(); annotation_dir.mkdir()
    repository = DatasetIndexRepository(tmp_path, image_dir, annotation_dir, "yolo")
    records = []
    for index in range(1200):
        path = image_dir / f"{index:05d}.jpg"
        records.append(IndexedImage(
            0, path, path.name, path.name, index + 1, index + 1,
        ))
    repository.upsert_batch(records)
    settings = ProjectSettings(
        image_dir=image_dir, annotation_dir=annotation_dir,
        annotation_format="yolo", dataset_task="yolo_detection",
    )
    worker = AutoLabelWorker(None, settings)

    iterator, total = worker._records_and_total()

    assert total == 1200
    loaded = list(iterator)
    assert len(loaded) == 1200
    assert loaded[0].path.name == "00000.jpg"
    assert loaded[-1].path.name == "01199.jpg"
