from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPointF

from src.models.annotation import Annotation, Keypoint, LabelPreset, ShapeType
from src.models.project import ProjectSettings
from src.services.annotation_service import AnnotationService


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 240), (40, 80, 120)).save(path, "JPEG")


def _settings(image_dir: Path, annotation_dir: Path, fmt: str, task: str) -> ProjectSettings:
    return ProjectSettings(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        annotation_format=fmt,
        dataset_task=task,
        label_presets=[LabelPreset("person", 0, "#00e5ff"), LabelPreset("car", 1, "#ffcc00")],
    )


def test_generated_dataset_matrix_round_trips_every_supported_format(tmp_path):
    """Build fresh datasets in tmp_path; never depend on checked-in data."""
    service = AnnotationService()
    cases = [
        ("yolo-detect", "yolo", "yolo_detection", Annotation(ShapeType.RECTANGLE, "person", [QPointF(20, 30), QPointF(120, 160)])),
        ("yolo-segment", "yolo", "yolo_segmentation", Annotation(ShapeType.POLYGON, "person", [QPointF(20, 30), QPointF(120, 30), QPointF(80, 160)])),
        ("yolo-obb", "yolo", "yolo_obb", Annotation(ShapeType.OBB, "car", [QPointF(20, 30), QPointF(120, 30), QPointF(130, 140), QPointF(30, 150)])),
        ("voc", "voc", "voc", Annotation(ShapeType.RECTANGLE, "person", [QPointF(20, 30), QPointF(120, 160)])),
    ]
    for name, fmt, task, annotation in cases:
        root = tmp_path / name
        image_dir, annotation_dir = root / "images", root / ("labels" if fmt == "yolo" else "Annotations")
        image = image_dir / "nested" / "sample.jpg"
        _image(image)
        settings = _settings(image_dir, annotation_dir, fmt, task)
        assert service.save(image, [annotation], annotation_dir / "nested" if fmt == "yolo" else annotation_dir, settings).ok
        loaded = service.load(image, annotation_dir / "nested" if fmt == "yolo" else annotation_dir, settings)
        assert loaded.error is None
        assert len(loaded.annotations) == 1
        assert loaded.annotations[0].shape_type == annotation.shape_type

    root = tmp_path / "yolo-pose"
    image_dir, annotation_dir = root / "images", root / "labels"
    image = image_dir / "pose.jpg"
    _image(image)
    settings = _settings(image_dir, annotation_dir, "yolo", "yolo_pose")
    pose = Annotation(
        ShapeType.KEYPOINT, "person", [QPointF(20, 30), QPointF(120, 160)],
        keypoints=[Keypoint("nose", QPointF(60, 70), 2), Keypoint("eye", QPointF(70, 80), 1)],
    )
    settings.keypoint_count = 2
    assert service.save(image, [pose], annotation_dir, settings).ok
    loaded = service.load(image, annotation_dir, settings)
    assert loaded.error is None and len(loaded.annotations) == 1
    assert len(loaded.annotations[0].keypoints) == 2

    root = tmp_path / "coco"
    image_dir, annotation_dir = root / "images", root / "annotations"
    image = image_dir / "coco.jpg"
    _image(image)
    settings = _settings(image_dir, annotation_dir, "coco", "coco")
    annotation = Annotation(ShapeType.RECTANGLE, "person", [QPointF(20, 30), QPointF(120, 160)])
    assert service.save(image, [annotation], annotation_dir, settings).ok
    loaded = service.load(image, annotation_dir, settings)
    assert loaded.error is None and len(loaded.annotations) == 1

    # Exercise the critical deletion path on the generated COCO dataset.
    assert service.save(image, [], annotation_dir, settings).ok
    assert service.load(image, annotation_dir, settings).annotations == []
