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
