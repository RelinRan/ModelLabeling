import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PIL import Image

from src.services.dataset_detector import DatasetDetector
from src.services.dataset_initializer import DatasetInitializer
from src.widgets.dataset_init_dialog import DatasetInitDialog


def _images(root: Path, folder: str = "", count: int = 2) -> Path:
    target = root / folder if folder else root
    target.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", (32, 24), (20, 20, 20)).save(target / f"img_{index}.jpg")
    return root


def test_initializer_creates_detectable_structure_for_every_task(tmp_path):
    expectations = {
        "yolo_detection": ("yolo", "yolo_detection"),
        "yolo_segmentation": ("yolo", "yolo_segmentation"),
        "yolo_pose": ("yolo", "yolo_pose"),
        "yolo_obb": ("yolo", "yolo_obb"),
        "voc": ("voc", ""),
        "coco": ("coco", ""),
    }
    for task, expected in expectations.items():
        root = _images(tmp_path / task)
        names = ["cat", "dog"]
        info = DatasetInitializer.initialize(root, task, names, keypoint_count=6)
        assert DatasetInitializer.verify(root) == expected, task
        # Images were not moved.
        assert (root / "img_0.jpg").is_file(), task
        assert info.image_dir == root, task
        if task == "voc":
            assert (root / "Annotations").is_dir()
        elif task == "coco":
            document = json.loads((root / "annotations" / "annotations.json").read_text())
            assert set(document) >= {"images", "annotations", "categories"}
        else:
            assert (root / "labels").is_dir()
            yaml_text = (root / "data.yaml").read_text()
            assert f"task: {DatasetInitializer._yaml_task(task)}" in yaml_text
            if task == "yolo_pose":
                assert "kpt_shape: [6, 3]" in yaml_text
            assert "names: [cat, dog]" in yaml_text
            assert (root / "classes.txt").read_text().splitlines() == ["cat", "dog"]


def test_initializer_uses_existing_images_subfolder_and_keeps_files(tmp_path):
    root = _images(tmp_path / "nested", folder="images", count=3)
    info = DatasetInitializer.initialize(root, "yolo_detection", None)
    assert info.image_dir == root / "images"
    assert (root / "images" / "img_2.jpg").is_file()
    assert DatasetInitializer.verify(root) == ("yolo", "yolo_detection")


def test_initializer_rejects_folder_without_images(tmp_path):
    (tmp_path / "empty").mkdir()
    try:
        DatasetInitializer.initialize(tmp_path / "empty", "yolo_detection")
        raise AssertionError("must reject an image-less folder")
    except ValueError:
        pass


def test_initializer_is_idempotent(tmp_path):
    root = _images(tmp_path / "again")
    first = DatasetInitializer.initialize(root, "yolo_pose", ["person"], 12)
    (root / "data.yaml").write_text("# user edited\n", encoding="utf-8")
    second = DatasetInitializer.initialize(root, "yolo_pose", ["person"], 12)
    assert not second.created or all(path not in first.created for path in second.created)
    assert (root / "data.yaml").read_text() == "# user edited\n"  # never overwritten


def test_create_in_workspace_copies_images_and_builds_structure(tmp_path):
    workspace = tmp_path / "ws"
    source = _images(tmp_path / "photos")  # plain folder with 2 images

    info = DatasetInitializer.create_in_workspace(
        workspace, "my-dataset", "yolo_detection", ["cat", "dog"], source_dir=source
    )
    assert info.root == workspace / "my-dataset"
    assert DatasetInitializer.verify(info.root) == ("yolo", "yolo_detection")
    # Images were copied into the dataset, not referenced in place.
    assert (info.root / "images" / "img_0.jpg").is_file()
    assert (source / "img_0.jpg").is_file()
    assert info.copied_images == 2
    assert (info.root / "classes.txt").read_text().splitlines() == ["cat", "dog"]

    # Duplicate names are rejected; invalid names are rejected.
    try:
        DatasetInitializer.create_in_workspace(workspace, "my-dataset", "voc", source_dir=source)
        raise AssertionError("duplicate name must be rejected")
    except ValueError as exc:
        assert "同名" in str(exc)
    try:
        DatasetInitializer.create_in_workspace(workspace, 'bad:name', "voc")
        raise AssertionError("invalid name must be rejected")
    except ValueError as exc:
        assert "字符" in str(exc)

    # A source folder with nested subfolders is flattened, collisions prefixed.
    nested = tmp_path / "nested-photos"
    (nested / "a").mkdir(parents=True)
    (nested / "b").mkdir()
    Image.new("RGB", (8, 8)).save(nested / "a" / "shot.jpg")
    Image.new("RGB", (8, 8)).save(nested / "b" / "shot.jpg")
    info2 = DatasetInitializer.create_in_workspace(workspace, "flattened", "voc", source_dir=nested)
    names = sorted(p.name for p in (info2.root / "images").glob("*.jpg"))
    assert len(names) == 2 and all("shot" in name for name in names)
    assert info2.copied_images == 2


def test_create_in_workspace_without_source_makes_empty_dataset(tmp_path):
    info = DatasetInitializer.create_in_workspace(tmp_path / "ws2", "empty-pose", "yolo_pose", ["person"], keypoint_count=6)
    assert info.copied_images == 0
    assert DatasetInitializer.verify(info.root) == ("yolo", "yolo_pose")
    assert "kpt_shape: [6, 3]" in (info.root / "data.yaml").read_text()


def test_create_in_workspace_imports_annotations_from_existing_dataset(tmp_path):
    # Build a YOLO source dataset with labels.
    source = _images(tmp_path / "src-yolo", count=2)
    (source / "labels").mkdir()
    (source / "labels" / "img_0.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (source / "classes.txt").write_text("ship\n", encoding="utf-8")

    info = DatasetInitializer.create_in_workspace(
        tmp_path / "ws3", "imported", "yolo_detection", None, source_dir=source
    )
    assert info.copied_images == 2
    assert info.imported_annotations == 1
    assert (info.root / "labels" / "img_0.txt").is_file()
    # Source annotation survives untouched.
    assert (source / "labels" / "img_0.txt").is_file()


def test_init_dialog_reports_state_and_initializes(tmp_path):
    app = QApplication.instance() or QApplication([])
    workspace = tmp_path / "dlg-ws"
    source = _images(tmp_path / "dlg-src")

    dialog = DatasetInitDialog(None, "zh_CN", str(workspace))
    dialog.name_edit.setText("fresh")
    dialog.source_edit.setText(str(source))
    assert "2 张图片" in dialog.status_label.text()

    results = []
    dialog.initialized.connect(lambda info: results.append(info))
    dialog.task_combo.setCurrentIndex(3)  # YOLO OBB
    dialog._create()
    assert results and results[0][0] == "created"
    info = results[0][1]
    assert DatasetInitializer.verify(info.root) == ("yolo", "yolo_obb")
    assert info.copied_images == 2

    # Duplicate name shows a clear message instead of creating anything.
    dialog2 = DatasetInitDialog(None, "zh_CN", str(workspace))
    dialog2.name_edit.setText("fresh")
    assert "已存在同名数据集" in dialog2.status_label.text()

    # Missing name is reported.
    dialog3 = DatasetInitDialog(None, "zh_CN", str(workspace))
    dialog3.source_edit.setText(str(source))
    assert "数据集名称" in dialog3.status_label.text()
