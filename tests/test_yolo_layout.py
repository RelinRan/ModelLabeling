"""Plain image folders are converted to the Ultralytics YOLO layout once."""
import os
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.models.annotation import LabelPreset
from src.services.dataset_detector import DatasetDetector
from src.services.yolo_layout import migrate_plain_folder, needs_standard_layout


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), "gray").save(path)
    return path


def _ultralytics_label_path(image: Path) -> Path:
    """The label path Ultralytics derives from an image path."""
    resolved = image.resolve().as_posix()
    return Path("/labels/".join(resolved.rsplit("/images/", 1)).rsplit(".", 1)[0] + ".txt")


def test_plain_yolo_folder_needs_the_standard_layout(tmp_path):
    root = tmp_path / "plain"
    root.mkdir()
    assert needs_standard_layout(root, root, root / "labels", "yolo")
    (root / "images").mkdir()
    assert not needs_standard_layout(root, root / "images", root / "labels", "yolo")
    assert not needs_standard_layout(root, root, root / "labels", "coco")
    assert not needs_standard_layout(root, root, root / "labels", "voc")
    assert not needs_standard_layout(None, None, None, "yolo")


def test_migration_moves_images_and_beside_labels(tmp_path):
    root = tmp_path / "photos"
    _image(root / "a.jpg")
    _image(root / "sub" / "b.png")
    (root / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "notes.txt").write_text("leave me", encoding="utf-8")

    migration = migrate_plain_folder(
        root, presets=[LabelPreset("person", 0, "#00e5ff")], task_name="yolo_segmentation",
    )

    assert migration.moved_images == 2
    assert migration.moved_labels == 1
    assert (root / "images" / "a.jpg").is_file()
    assert (root / "images" / "sub" / "b.png").is_file()
    assert (root / "labels" / "a.txt").is_file()
    assert not (root / "a.jpg").exists()
    assert not (root / "a.txt").exists()
    # Non-image, non-label files are left where they were.
    assert (root / "notes.txt").read_text(encoding="utf-8") == "leave me"

    # Every image now resolves to its label through the trainer's own rule.
    assert _ultralytics_label_path(root / "images" / "a.jpg") == (root / "labels" / "a.txt").resolve()
    assert not _ultralytics_label_path(root / "images" / "sub" / "b.png").exists()

    document = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    assert document["path"] == root.resolve().as_posix()
    assert document["train"] == "images"
    assert document["val"] == "images"
    assert document["task"] == "segment"
    assert document["names"] == {0: "person"}
    assert (root / "classes.txt").read_text(encoding="utf-8").split() == ["person"]

    detected = DatasetDetector.detect(root)
    assert detected.format_name == "yolo"
    assert detected.task_name == "yolo_segmentation"
    assert detected.image_dir.resolve() == (root / "images").resolve()
    assert detected.annotation_dir.resolve() == (root / "labels").resolve()


@pytest.mark.parametrize("task_name, expected", [
    ("yolo_detection", "detect"),
    ("yolo_segmentation", "segment"),
    ("yolo_pose", "pose"),
    ("yolo_obb", "obb"),
])
def test_migrated_yaml_declares_the_selected_task(tmp_path, task_name, expected):
    root = tmp_path / task_name
    _image(root / "a.jpg")
    migrate_plain_folder(root, presets=[LabelPreset("object", 0, "#00e5ff")], task_name=task_name)
    assert yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["task"] == expected


def test_existing_labels_directory_is_kept(tmp_path):
    root = tmp_path / "photos"
    _image(root / "a.jpg")
    (root / "labels").mkdir()
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    migration = migrate_plain_folder(root, presets=[LabelPreset("person", 0, "#00e5ff")])

    assert migration.moved_images == 1
    assert migration.moved_labels == 0
    assert (root / "images" / "a.jpg").is_file()
    assert (root / "labels" / "a.txt").is_file()


def test_conflicting_destination_aborts_before_any_move(tmp_path):
    root = tmp_path / "photos"
    _image(root / "a.jpg")
    _image(root / "images" / "a.jpg")  # an earlier, unrelated file owns the target

    with pytest.raises(ValueError):
        migrate_plain_folder(root)

    assert (root / "a.jpg").is_file(), "the source must not be moved on a conflict"
    assert (root / "images" / "a.jpg").is_file()


def test_failed_move_rolls_back_completed_moves(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    _image(root / "a.jpg")
    _image(root / "b.jpg")
    real_replace = os.replace
    calls = {"count": 0}

    def flaky_replace(source, target):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated failure")
        return real_replace(source, target)

    monkeypatch.setattr("src.services.yolo_layout.os.replace", flaky_replace)
    with pytest.raises(OSError):
        migrate_plain_folder(root)

    assert (root / "a.jpg").is_file()
    assert (root / "b.jpg").is_file()
    assert not (root / "images" / "a.jpg").exists()


def test_existing_class_table_wins_over_app_presets(tmp_path):
    root = tmp_path / "photos"
    _image(root / "a.jpg")
    (root / "a.txt").write_text("1 0.5 0.5 0.2 0.2 0.4 0.4\n", encoding="utf-8")
    (root / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")

    migrate_plain_folder(
        root, presets=[LabelPreset("person", 0, "#00e5ff"), LabelPreset("car", 1, "#ff0000")],
        task_name="yolo_segmentation",
    )

    document = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    assert document["names"] == {0: "cat", 1: "dog"}
    assert (root / "classes.txt").read_text(encoding="utf-8").split() == ["cat", "dog"]


def test_class_table_inside_labels_is_kept(tmp_path):
    root = tmp_path / "photos"
    _image(root / "a.jpg")
    (root / "labels").mkdir()
    (root / "labels" / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")

    migrate_plain_folder(root, presets=[LabelPreset("person", 0, "#00e5ff")])

    document = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    assert document["names"] == {0: "cat", 1: "dog"}
    assert not (root / "classes.txt").exists()


def test_existing_data_yaml_names_are_kept(tmp_path):
    root = tmp_path / "photos"
    _image(root / "a.jpg")
    (root / "data.yaml").write_text(
        yaml.safe_dump({"names": ["cat", "dog"], "train": "old/path"}), encoding="utf-8",
    )

    migrate_plain_folder(root, presets=[LabelPreset("person", 0, "#00e5ff")])

    document = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    assert document["names"] == {0: "cat", 1: "dog"}
    assert document["train"] == "images", "the folder is the dataset root now"


def test_data_yaml_keeps_pose_schema_metadata(tmp_path):
    root = tmp_path / "photos"
    _image(root / "a.jpg")
    (root / "data.yaml").write_text(
        yaml.safe_dump({"kpt_shape": [2, 3], "kpt_names": ["left", "right"]}),
        encoding="utf-8",
    )
    migrate_plain_folder(root, presets=[LabelPreset("person", 0, "#00e5ff")], task_name="yolo_pose")
    document = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    assert document["task"] == "pose"
    assert document["kpt_shape"] == [2, 3]
    assert document["kpt_names"] == ["left", "right"]