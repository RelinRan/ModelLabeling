import json
from pathlib import Path

from src.services.dataset_index import DatasetIndexRepository


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")


def test_duplicate_basenames_have_distinct_positions(tmp_path):
    image_dir = tmp_path / "images"
    labels = tmp_path / "labels"
    labels.mkdir()
    first = image_dir / "train" / "same.jpg"
    second = image_dir / "val" / "same.jpg"
    _image(first); _image(second)
    repository = DatasetIndexRepository(tmp_path, image_dir, labels, "yolo")
    repository.upsert_batch([item for batch in repository.scan_paths(batch_size=10) for item in batch])

    assert {repository.position(first), repository.position(second)} == {0, 1}
    assert [item.path for page in repository.iter_pages(1) for item in page] == [first, second]


def test_index_identity_includes_image_and_annotation_directories(tmp_path):
    image_a = tmp_path / "a" / "sample.jpg"
    image_b = tmp_path / "b" / "sample.jpg"
    _image(image_a)
    _image(image_b)
    first = DatasetIndexRepository(tmp_path, image_a.parent, tmp_path / "labels", "yolo")
    second = DatasetIndexRepository(tmp_path, image_b.parent, tmp_path / "labels", "yolo")
    for repository in (first, second):
        for batch in repository.scan_paths():
            repository.upsert_batch(batch)

    assert first.path != second.path
    assert [item.path for item in first.get_page(0, 10)] == [image_a]
    assert [item.path for item in second.get_page(0, 10)] == [image_b]
    equivalent = DatasetIndexRepository(tmp_path / "different-entry", image_a.parent, tmp_path / "labels", "yolo")
    assert equivalent.path == first.path
    changed_labels = DatasetIndexRepository(tmp_path, image_a.parent, tmp_path / "other-labels", "yolo")
    assert changed_labels.path != first.path


def test_coco_status_and_labels_use_relative_path_without_basename_merging(tmp_path):
    image_dir = tmp_path / "images"
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    train = image_dir / "train" / "same.jpg"
    val = image_dir / "val" / "same.jpg"
    empty = image_dir / "test" / "empty.jpg"
    for path in (train, val, empty):
        _image(path)
    document = {
        "images": [
            {"id": 1, "file_name": "train/same.jpg"},
            {"id": 2, "file_name": "val/same.jpg"},
            {"id": 3, "file_name": "test/empty.jpg"},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "vehicle"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1},
            {"id": 2, "image_id": 2, "category_id": 2},
            {"id": 3, "image_id": 3, "category_id": 999},
        ],
    }
    json_path = annotation_dir / "annotations.json"
    json_path.write_text(json.dumps(document), encoding="utf-8")
    repository = DatasetIndexRepository(tmp_path, image_dir, annotation_dir, "coco")
    records = [item for batch in repository.scan_paths(batch_size=10) for item in batch]
    repository.upsert_batch(records)
    by_relative = {item.relative_path.replace("\\", "/"): item for item in records}

    assert by_relative["train/same.jpg"].annotation_labels == ("person",)
    assert by_relative["val/same.jpg"].annotation_labels == ("vehicle",)
    assert by_relative["test/empty.jpg"].annotation_status == "present"
    assert by_relative["test/empty.jpg"].annotation_labels == ()
    assert repository.count(status="labeled") == 3
    assert repository.count(status="unlabeled") == 0
    assert [item.relative_path.replace("\\", "/") for item in repository.get_page(0, 10, label="vehicle")] == ["val/same.jpg"]


def test_label_filter_matches_the_complete_name_not_a_substring(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    paths = [image_dir / f"{name}.jpg" for name in ("car", "cart", "racecar")]
    for path in paths:
        _image(path)
    repository = DatasetIndexRepository(tmp_path, image_dir, label_dir, "yolo")
    records = [item for batch in repository.scan_paths(batch_size=10) for item in batch]
    repository.upsert_batch(records)
    repository.update_annotation(paths[0], label_dir / "car.txt", ["car"])
    repository.update_annotation(paths[1], label_dir / "cart.txt", ["cart"])
    repository.update_annotation(paths[2], label_dir / "racecar.txt", ["racecar"])

    assert [item.path.name for item in repository.get_page(0, 10, label="car")] == ["car.jpg"]
    assert [item.path.name for item in repository.get_page(0, 10, label="CAR")] == ["car.jpg"]


def test_empty_annotation_files_are_unlabeled_and_save_updates_immediately(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image = image_dir / "sample.jpg"
    label = label_dir / "sample.txt"
    _image(image)
    label_dir.mkdir()
    label.write_text("", encoding="utf-8")
    repository = DatasetIndexRepository(tmp_path, image_dir, label_dir, "yolo")
    records = [item for batch in repository.scan_paths(batch_size=10, label_names=["person"]) for item in batch]
    repository.upsert_batch(records)

    assert records[0].annotation_status == "missing"
    repository.update_annotation(image, label, ["person"])
    assert repository.count(status="labeled") == 1
    repository.update_annotation(image, label, [])
    assert repository.count(status="labeled") == 0
    assert repository.count(status="unlabeled") == 1


def test_empty_voc_document_is_unlabeled(tmp_path):
    image_dir = tmp_path / "images"
    annotation_dir = tmp_path / "Annotations"
    image = image_dir / "sample.jpg"
    _image(image)
    annotation_dir.mkdir()
    (annotation_dir / "sample.xml").write_text(
        "<annotation><filename>sample.jpg</filename></annotation>", encoding="utf-8",
    )
    repository = DatasetIndexRepository(tmp_path, image_dir, annotation_dir, "voc")

    records = [item for batch in repository.scan_paths(batch_size=10) for item in batch]

    assert records[0].annotation_status == "missing"


def test_batch_upsert_replaces_labels_with_empty_for_large_batches(tmp_path):
    image_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images = []
    for index in range(950):
        image = image_dir / f"{index:04d}.jpg"
        _image(image)
        images.append(image)
    repository = DatasetIndexRepository(tmp_path, image_dir, labels_dir, "yolo")
    records = [item for batch in repository.scan_paths(batch_size=1000) for item in batch]
    first_write = [
        type(record)(**{**record.__dict__, "annotation_labels": ("person",)})
        for record in records
    ]
    repository.upsert_batch(first_write)
    assert repository.count(label="person") == 950

    second_write = [
        type(record)(**{**record.__dict__, "annotation_labels": ("vehicle",) if record.path == images[0] else ()})
        for record in records
    ]
    repository.upsert_batch(second_write)
    assert repository.count(label="person") == 0
    assert repository.count(label="vehicle") == 1


def test_statistics_snapshot_aggregates_index_and_reflects_annotation_edits(tmp_path):
    image_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images = [image_dir / f"{index}.jpg" for index in range(3)]
    for image in images:
        _image(image)
    repository = DatasetIndexRepository(tmp_path, image_dir, labels_dir, "yolo")
    repository.upsert_batch([item for batch in repository.scan_paths(batch_size=10) for item in batch])
    repository.update_annotation(images[0], labels_dir / "0.txt", ["person", "vehicle"])
    repository.update_annotation(images[1], labels_dir / "1.txt", ["person"])

    snapshot = repository.statistics_snapshot()
    assert snapshot["total_images"] == 3
    assert snapshot["labeled_images"] == 2
    assert round(snapshot["percentage"], 4) == round(200 / 3, 4)
    assert snapshot["total_labels"] == 3
    assert snapshot["label_counts"] == {"person": 2, "vehicle": 1}
    repository.update_annotation(images[0], labels_dir / "0.txt", ["vehicle"])
    snapshot = repository.statistics_snapshot()
    assert snapshot["labeled_images"] == 2
    assert snapshot["total_labels"] == 2
    assert snapshot["label_counts"] == {"person": 1, "vehicle": 1}


def test_scanned_page_and_snapshot_exclude_stale_cached_rows(tmp_path):
    image_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    first, stale = image_dir / "first.jpg", image_dir / "stale.jpg"
    _image(first)
    _image(stale)
    repository = DatasetIndexRepository(tmp_path, image_dir, labels_dir, "yolo")
    old_records = [item for batch in repository.scan_paths(batch_size=10) for item in batch]
    repository.upsert_batch(old_records)
    repository.set_complete(True)
    stale.unlink()

    repository.set_complete(False)
    fresh_records = [item for batch in repository.scan_paths(batch_size=10, first_batch_size=1) for item in batch]
    repository.upsert_batch(fresh_records)

    assert repository.scanned_count() == 1
    assert [item.path for item in repository.get_scanned_page(0, 10)] == [first]
    assert repository.statistics_snapshot(scanned_only=True)["total_images"] == 1
