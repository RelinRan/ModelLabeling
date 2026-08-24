import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPointF

from src.models.annotation import Annotation, Keypoint, LabelPreset, ShapeType
from src.models.project import ProjectSettings
from src.services.annotation_service import AnnotationService
from src.services.coco_store import CocoAnnotationStore


def _sample_image(path: Path) -> None:
    Image.new("RGB", (640, 480), (20, 20, 20)).save(path)


def _settings(root: Path, fmt: str, task: str, presets: list[LabelPreset]) -> ProjectSettings:
    return ProjectSettings(
        image_dir=root,
        annotation_dir=root,
        annotation_format=fmt,
        dataset_task=task,
        label_presets=presets,
    )


def test_official_shapes_round_trip(tmp_path):
    image = tmp_path / "sample.jpg"
    _sample_image(image)
    presets = [LabelPreset("person", 0, "#00e5ff")]
    service = AnnotationService()
    cases = [
        ("voc", "voc", Annotation(ShapeType.SQUARE, "person", [QPointF(10, 20), QPointF(110, 120)])),
        ("yolo", "yolo_detection", Annotation(ShapeType.RECTANGLE, "person", [QPointF(10, 20), QPointF(110, 120)])),
        ("yolo", "yolo_segmentation", Annotation(ShapeType.POLYGON, "person", [QPointF(10, 20), QPointF(110, 20), QPointF(60, 120)])),
    ]
    for index, (fmt, task, annotation) in enumerate(cases):
        directory = tmp_path / f"case-{index}"
        directory.mkdir()
        settings = _settings(directory, fmt, task, presets)
        result = service.save(image, [annotation], directory, settings)
        loaded = service.load(image, directory, settings)
        assert result.ok and loaded.error is None
        assert len(loaded.annotations) == 1
        assert loaded.annotations[0].label == "person"


def test_coco_and_yolo_pose_round_trip(tmp_path):
    image = tmp_path / "sample.jpg"
    _sample_image(image)
    presets = [LabelPreset("person", 0, "#00e5ff")]
    keypoints = [Keypoint("nose", QPointF(50, 50), 2), Keypoint("eye", QPointF(60, 55), 1)]
    annotation = Annotation(ShapeType.KEYPOINT, "person", [QPointF(30, 30), QPointF(100, 100)], keypoints=keypoints)
    service = AnnotationService()
    for index, (fmt, task) in enumerate((("coco", "coco"), ("yolo", "yolo_pose"))):
        directory = tmp_path / f"pose-{index}"
        directory.mkdir()
        settings = _settings(directory, fmt, task, presets)
        assert service.save(image, [annotation], directory, settings).ok
        loaded = service.load(image, directory, settings)
        assert loaded.error is None and len(loaded.annotations) == 1
        assert len(loaded.annotations[0].keypoints) == 2


def test_coco_store_preserves_document(tmp_path):
    directory = tmp_path / "annotations"
    store = CocoAnnotationStore(directory)
    document = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 640, "height": 480}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]}],
        "categories": [{"id": 1, "name": "person", "supercategory": "object"}],
        "info": {"description": "test"},
        "licenses": [],
    }
    store.replace_document(document)
    assert store.read_document() == document
