import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from PySide6.QtCore import QPointF

from src.models.annotation import Annotation, LabelPreset, ShapeType
from src.models.project import ProjectSettings
from src.services.annotation_service import AnnotationService
from src.services.coco_store import CocoAnnotationStore
from src.services.conversion_service import ConversionOptions, ConversionService
from src.services.dataset_index import DatasetIndexRepository


def coco_dataset(root, names=("a/same.jpg", "b/same.jpg")):
    images = root / "images"
    labels = root / "annotations"
    labels.mkdir(parents=True)
    for name in names:
        path = images / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48)).save(path)
    document = {
        "images": [dict(id=i, file_name=name, width=64, height=48) for i, name in enumerate(names, 1)],
        "categories": [dict(id=1, name="old"), dict(id=2, name="new")],
        "annotations": [dict(id=i, image_id=i, category_id=1, bbox=[1, 2, 10, 12]) for i, _ in enumerate(names, 1)],
    }
    path = labels / "annotations.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    settings = ProjectSettings(image_dir=images, annotation_dir=labels, annotation_format="coco", dataset_task="coco")
    return settings, path, document


def test_coco_first_save_preserves_other_images_and_nested_identity(tmp_path):
    settings, path, _ = coco_dataset(tmp_path)
    service = AnnotationService()
    image = settings.image_dir / "a/same.jpg"
    box = Annotation(ShapeType.RECTANGLE, "new", [QPointF(3, 4), QPointF(20, 24)])
    assert service.save(image, [box], settings.annotation_dir, settings).ok
    assert [a.label for a in service.load(image, settings.annotation_dir, settings).annotations] == ["new"]
    assert [a.label for a in service.load(settings.image_dir / "b/same.jpg", settings.annotation_dir, settings).annotations] == ["old"]
    service.export_coco(settings.annotation_dir)
    assert {i["file_name"] for i in json.loads(path.read_text())["images"]} == {"a/same.jpg", "b/same.jpg"}


def test_coco_external_json_change_refreshes_clean_cache(tmp_path):
    settings, path, document = coco_dataset(tmp_path)
    service = AnnotationService()
    image = settings.image_dir / "a/same.jpg"
    assert service.load(image, settings.annotation_dir, settings).annotations[0].label == "old"
    document["categories"][0]["name"] = "external-new-name"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert service.load(image, settings.annotation_dir, settings).annotations[0].label == "external-new-name"
    service.export_coco(settings.annotation_dir)
    assert json.loads(path.read_text())["categories"][0]["name"] == "external-new-name"


def test_coco_external_change_does_not_overwrite_pending_edits(tmp_path):
    settings, path, document = coco_dataset(tmp_path)
    service = AnnotationService()
    image = settings.image_dir / "a/same.jpg"
    box = Annotation(ShapeType.RECTANGLE, "local", [QPointF(3, 4), QPointF(20, 24)])
    assert service.save(image, [box], settings.annotation_dir, settings).ok
    document["categories"][0]["name"] = "external"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="externally"):
        service.export_coco(settings.annotation_dir)
    assert json.loads(path.read_text())["categories"][0]["name"] == "external"
    assert CocoAnnotationStore(settings.annotation_dir).is_dirty()


def test_coco_index_keeps_root_image_with_same_basename(tmp_path):
    settings, _, _ = coco_dataset(tmp_path, ("same.jpg", "nested/same.jpg"))
    service = AnnotationService()
    index = service.build_index(settings.annotation_dir, "coco")
    assert index.coco_images["same.jpg"]["id"] == 1
    assert index.coco_images["nested/same.jpg"]["id"] == 2


def test_coco_preview_uses_single_image_query(tmp_path, monkeypatch):
    settings, _, _ = coco_dataset(tmp_path)
    service = AnnotationService()
    service._ensure_coco_store(settings.annotation_dir)
    monkeypatch.setattr(CocoAnnotationStore, "read_document", lambda *_: pytest.fail("Full document read during preview"))
    loaded = service.load(settings.image_dir / "a/same.jpg", settings.annotation_dir, settings)
    assert not loaded.error and len(loaded.annotations) == 1


def test_non_overwrite_coco_conversion_preserves_existing_dataset(tmp_path):
    source, _, _ = coco_dataset(tmp_path / "source", ("new.jpg",))
    target, path, _ = coco_dataset(tmp_path / "target", ("existing.jpg",))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="not empty"):
        ConversionService().convert(ConversionOptions("coco", tmp_path / "source", "coco", tmp_path / "target", []))
    assert path.read_bytes() == original
    assert (target.image_dir / "existing.jpg").exists()


def test_yolo_mapping_change_invalidates_cached_labels(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir(); labels.mkdir()
    (images / "one.jpg").write_bytes(b"image")
    (labels / "one.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    repository = DatasetIndexRepository(tmp_path, images, labels, "yolo")
    for names in (["old"], ["new"]):
        for batch in repository.scan_paths(label_names=names):
            repository.upsert_batch(batch)
    assert repository.count(label="new") == 1
    assert repository.count(label="old") == 0


def test_statistics_failure_is_not_marked_complete():
    from src.widgets.main_window import MainWindow
    owner = SimpleNamespace(
        dataset_session_id="current", _dataset_statistics={"total_images": 1},
        settings=SimpleNamespace(language="en_US"), _statistics_completed=False,
        status_progress_host=SimpleNamespace(setVisible=lambda _: None), isVisible=lambda: False,
    )
    MainWindow._dataset_statistics_failed(owner, "broken", "old-session")
    assert owner._dataset_statistics is not None
    MainWindow._dataset_statistics_failed(owner, "broken", "current")
    assert owner._dataset_statistics is None
    assert not owner._statistics_completed and owner._statistics_error == "broken"


@pytest.mark.parametrize("output_format", ["yolo", "voc", "coco"])
def test_conversion_can_cancel_at_initial_progress(tmp_path, output_format, monkeypatch):
    coco_dataset(tmp_path / "source", ("one.jpg",))
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.bin"
    sentinel.write_bytes(b"original")
    progress = []
    monkeypatch.setattr(
        ConversionService, "_complete_presets",
        lambda *_: pytest.fail("Category discovery ran after cancellation"),
    )
    report = ConversionService().convert(
        ConversionOptions("coco", tmp_path / "source", output_format, output, [], overwrite=True),
        progress_callback=lambda current, total: progress.append((current, total)),
        cancel_callback=lambda: bool(progress),
    )
    assert progress == [(0, 1)]
    assert report.cancelled and report.succeeded == 0 and report.failed == 0
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"original"
    assert not list(tmp_path.glob(".output.staging-*"))
