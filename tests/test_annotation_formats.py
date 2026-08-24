import os
import json
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPointF

from src.models.annotation import Annotation, Keypoint, LabelPreset, ShapeType
from src.models.project import ProjectSettings
from src.services.annotation_service import AnnotationService
from src.services.coco_store import CocoAnnotationStore
from src.services.conversion_service import ConversionOptions, ConversionService
from src.services.format_capabilities import UnsupportedAnnotationError, validate_annotations


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


def test_coco_single_image_transaction_and_checkpoint(tmp_path):
    directory = tmp_path / "annotations"
    store = CocoAnnotationStore(directory)
    store.replace_document({
        "images": [{"id": 1, "file_name": "old.jpg", "width": 10, "height": 10}],
        "annotations": [],
        "categories": [{"id": 1, "name": "person", "supercategory": "object"}],
        "info": {}, "licenses": [],
    })
    store.upsert_image(
        "new.jpg", 640, 480,
        [{"name": "person", "supercategory": "object"}],
        [{"category_name": "person", "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0, "segmentation": []}],
    )
    assert store.is_dirty()
    document = store.read_document()
    assert len(document["images"]) == 2
    assert len(document["annotations"]) == 1
    output = store.export_json()
    assert output.exists() and not store.is_dirty()
    assert json.loads(output.read_text(encoding="utf-8")) == document


def test_yolo_pose_rejects_mixed_keypoint_schema():
    first = Annotation(
        ShapeType.KEYPOINT, "person", [QPointF(0, 0), QPointF(10, 10)],
        keypoints=[Keypoint("nose", QPointF(1, 1), 2), Keypoint("eye", QPointF(2, 2), 2)],
    )
    second = Annotation(
        ShapeType.KEYPOINT, "person", [QPointF(0, 0), QPointF(10, 10)],
        keypoints=[Keypoint("nose", QPointF(1, 1), 2), Keypoint("ear", QPointF(2, 2), 2)],
    )
    with pytest.raises(UnsupportedAnnotationError):
        validate_annotations([first, second], "yolo", "yolo_pose")


def test_yolo_pose_load_validates_data_yaml_shape(tmp_path):
    image = tmp_path / "images" / "sample.jpg"
    labels = tmp_path / "labels"
    image.parent.mkdir(); labels.mkdir()
    _sample_image(image)
    (tmp_path / "data.yaml").write_text("kpt_shape: [3, 3]\n", encoding="utf-8")
    (labels / "sample.txt").write_text(
        "0 0.5 0.5 0.5 0.5 0.4 0.4 2 0.6 0.6 2\n", encoding="utf-8"
    )
    settings = _settings(tmp_path, "yolo", "yolo_pose", [LabelPreset("person", 0, "#00e5ff")])
    result = AnnotationService().load(image, labels, settings)
    assert result.error and "expected 3 keypoints" in result.error


def test_yolo_pose_load_accepts_matching_data_yaml_shape(tmp_path):
    image = tmp_path / "images" / "sample.jpg"
    labels = tmp_path / "labels"
    image.parent.mkdir(); labels.mkdir()
    _sample_image(image)
    (tmp_path / "data.yaml").write_text(
        "kpt_shape: [2, 3]\nkpt_names:\n  0: [nose, eye]\n",
        encoding="utf-8",
    )
    (labels / "sample.txt").write_text(
        "0 0.5 0.5 0.5 0.5 0.4 0.4 2 0.6 0.6 2\n", encoding="utf-8"
    )
    settings = _settings(tmp_path, "yolo", "yolo_pose", [LabelPreset("person", 0, "#00e5ff")])
    result = AnnotationService().load(image, labels, settings)
    assert not result.error
    assert len(result.annotations) == 1
    assert len(result.annotations[0].keypoints) == 2
    assert [item.name for item in result.annotations[0].keypoints] == ["nose", "eye"]


def test_coco_keypoints_convert_to_official_yolo_pose(tmp_path):
    source = tmp_path / "source"
    image = source / "images" / "sample.jpg"
    annotations_dir = source / "annotations"
    image.parent.mkdir(parents=True)
    annotations_dir.mkdir()
    _sample_image(image)
    presets = [LabelPreset("person", 0, "#00e5ff")]
    annotation = Annotation(
        ShapeType.KEYPOINT,
        "person",
        [QPointF(30, 30), QPointF(100, 100)],
        keypoints=[
            Keypoint("nose", QPointF(50, 50), 2),
            Keypoint("eye", QPointF(60, 55), 1),
        ],
    )
    AnnotationService().save_coco_batch([(image, [annotation])], annotations_dir, presets)

    output = tmp_path / "output"
    report = ConversionService().convert(ConversionOptions(
        source_format="coco",
        source_path=source,
        output_format="yolo",
        output_path=output,
        presets=presets,
        overwrite=True,
        source_task="coco",
        output_task="yolo_pose",
    ))

    assert report.succeeded == 1 and report.failed == 0
    yaml_text = (output / "data.yaml").read_text(encoding="utf-8")
    assert "kpt_shape: [2, 3]" in yaml_text
    assert "[nose, eye]" in yaml_text
    values = (output / "labels" / "sample.txt").read_text(encoding="utf-8").split()
    assert len(values) == 5 + 2 * 3


def test_coco_keypoints_take_priority_over_segmentation(tmp_path):
    image = tmp_path / "images" / "sample.jpg"
    annotations_dir = tmp_path / "annotations"
    image.parent.mkdir()
    annotations_dir.mkdir()
    _sample_image(image)
    document = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 640, "height": 480}],
        "categories": [{"id": 1, "name": "person", "keypoints": ["nose", "eye"], "skeleton": []}],
        "annotations": [{
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [30, 30, 70, 70],
            "segmentation": [[30, 30, 100, 30, 100, 100, 30, 100]],
            "keypoints": [50, 50, 2, 60, 55, 1],
            "num_keypoints": 2,
            "area": 4900,
            "iscrowd": 0,
        }],
    }
    (annotations_dir / "annotations.json").write_text(json.dumps(document), encoding="utf-8")
    settings = _settings(tmp_path, "coco", "coco", [LabelPreset("person", 0, "#00e5ff")])

    result = AnnotationService().load(image, annotations_dir, settings)

    assert not result.error
    assert len(result.annotations) == 1
    assert result.annotations[0].shape_type == ShapeType.KEYPOINT
    assert [item.name for item in result.annotations[0].keypoints] == ["nose", "eye"]


def test_coco_store_rejects_conflicting_category_keypoint_schema(tmp_path):
    store = CocoAnnotationStore(tmp_path / "annotations")
    store.upsert_image(
        "first.jpg", 100, 100,
        [{"name": "person", "keypoints": ["nose", "eye"]}],
        [],
    )

    with pytest.raises(ValueError, match="schema conflicts"):
        store.upsert_image(
            "second.jpg", 100, 100,
            [{"name": "person", "keypoints": ["nose", "ear"]}],
            [],
        )

    document = store.read_document()
    assert len(document["images"]) == 1
    assert document["categories"][0]["keypoints"] == ["nose", "eye"]


def test_coco_multipart_polygon_round_trip(tmp_path):
    image = tmp_path / "sample.jpg"
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    _sample_image(image)
    parts = [
        [QPointF(10, 10), QPointF(30, 10), QPointF(20, 30)],
        [QPointF(100, 100), QPointF(130, 100), QPointF(115, 130)],
    ]
    annotation = Annotation(
        ShapeType.POLYGON, "person", parts[0], polygon_parts=parts,
    )
    settings = _settings(tmp_path, "coco", "coco", [LabelPreset("person", 0, "#00e5ff")])
    service = AnnotationService()

    assert service.save(image, [annotation], annotations_dir, settings).ok
    loaded = service.load(image, annotations_dir, settings)

    assert not loaded.error
    assert len(loaded.annotations) == 1
    assert len(loaded.annotations[0].polygon_parts) == 2
    document = CocoAnnotationStore(annotations_dir).read_document()
    assert len(document["annotations"][0]["segmentation"]) == 2
    assert document["annotations"][0]["area"] == pytest.approx(650.0)


def test_yolo_segmentation_rejects_multipart_polygon():
    parts = [
        [QPointF(10, 10), QPointF(30, 10), QPointF(20, 30)],
        [QPointF(100, 100), QPointF(130, 100), QPointF(115, 130)],
    ]
    annotation = Annotation(
        ShapeType.POLYGON, "person", parts[0], polygon_parts=parts,
    )

    with pytest.raises(UnsupportedAnnotationError, match="multipart"):
        validate_annotations([annotation], "yolo", "yolo_segmentation")


def test_coco_rle_mask_is_not_silently_loaded_as_rectangle(tmp_path):
    image = tmp_path / "images" / "sample.jpg"
    annotations_dir = tmp_path / "annotations"
    image.parent.mkdir()
    annotations_dir.mkdir()
    _sample_image(image)
    document = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 640, "height": 480}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [{
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [30, 30, 70, 70],
            "segmentation": {"size": [480, 640], "counts": "compressed-rle"},
            "area": 4900,
            "iscrowd": 1,
        }],
    }
    (annotations_dir / "annotations.json").write_text(json.dumps(document), encoding="utf-8")
    settings = _settings(tmp_path, "coco", "coco", [LabelPreset("person", 0, "#00e5ff")])

    result = AnnotationService().load(image, annotations_dir, settings)

    assert not result.annotations
    assert result.error and "RLE mask" in result.error


def test_yolo_split_layout_reads_and_saves_matching_annotation_branch(tmp_path):
    image_dir = tmp_path / "images"
    annotation_dir = tmp_path / "labels"
    train_image = image_dir / "train" / "same.jpg"
    val_image = image_dir / "val" / "same.jpg"
    (annotation_dir / "train").mkdir(parents=True)
    (annotation_dir / "val").mkdir(parents=True)
    train_image.parent.mkdir(parents=True)
    val_image.parent.mkdir(parents=True)
    _sample_image(train_image)
    _sample_image(val_image)
    (annotation_dir / "train" / "same.txt").write_text(
        "0 0.25 0.25 0.2 0.2\n", encoding="utf-8",
    )
    (annotation_dir / "val" / "same.txt").write_text(
        "0 0.75 0.75 0.2 0.2\n", encoding="utf-8",
    )
    settings = ProjectSettings(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        annotation_format="yolo",
        dataset_task="yolo_detection",
        label_presets=[LabelPreset("person", 0, "#00e5ff")],
    )
    service = AnnotationService()

    train = service.load(train_image, annotation_dir, settings)
    val = service.load(val_image, annotation_dir, settings)

    assert not train.error and not val.error
    assert train.annotations[0].points[0].x() < val.annotations[0].points[0].x()
    replacement = Annotation(
        ShapeType.RECTANGLE, "person", [QPointF(100, 100), QPointF(200, 200)],
    )
    assert service.save(val_image, [replacement], annotation_dir, settings).ok
    assert "0.234375" in (annotation_dir / "val" / "same.txt").read_text(encoding="utf-8")
    assert "0.25 0.25" in (annotation_dir / "train" / "same.txt").read_text(encoding="utf-8")


def test_voc_nested_layout_saves_annotation_beside_matching_branch(tmp_path):
    image_dir = tmp_path / "images"
    annotation_dir = tmp_path / "Annotations"
    image = image_dir / "train" / "sample.jpg"
    image.parent.mkdir(parents=True)
    annotation_dir.mkdir()
    _sample_image(image)
    settings = ProjectSettings(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        annotation_format="voc",
        dataset_task="voc",
        label_presets=[LabelPreset("person", 0, "#00e5ff")],
    )
    annotation = Annotation(
        ShapeType.RECTANGLE, "person", [QPointF(10, 20), QPointF(110, 120)],
    )

    result = AnnotationService().save(image, [annotation], annotation_dir, settings)

    assert result.ok
    assert (annotation_dir / "train" / "sample.xml").exists()


def test_invalid_polygon_and_nonfinite_coordinates_are_rejected():
    with pytest.raises(ValueError, match="at least three"):
        Annotation(
            ShapeType.POLYGON, "person", [QPointF(0, 0), QPointF(10, 10)],
        )
    with pytest.raises(ValueError, match="finite"):
        Annotation(
            ShapeType.RECTANGLE, "person", [QPointF(0, 0), QPointF(float("nan"), 10)],
        )
    with pytest.raises(ValueError, match="finite"):
        Keypoint("nose", QPointF(float("inf"), 0), 2)


def test_yolo_pose_rejects_fractional_visibility(tmp_path):
    image = tmp_path / "images" / "sample.jpg"
    labels = tmp_path / "labels"
    image.parent.mkdir(); labels.mkdir()
    _sample_image(image)
    (tmp_path / "data.yaml").write_text(
        "kpt_shape: [1, 3]\nnames: [person]\n", encoding="utf-8",
    )
    (labels / "sample.txt").write_text(
        "0 0.5 0.5 0.5 0.5 0.4 0.4 1.6\n", encoding="utf-8",
    )
    settings = ProjectSettings(
        image_dir=tmp_path / "images", annotation_dir=labels,
        annotation_format="yolo", dataset_task="yolo_pose",
        label_presets=[LabelPreset("person", 0, "#00e5ff")],
    )

    result = AnnotationService().load(image, labels, settings)

    assert result.error and "visibility must be 0, 1, or 2" in result.error
