from pathlib import Path

from src.models.project import ProjectSettings
from src.services.dataset_detector import DatasetDetector
from src.services.workers import DatasetScanWorker
from src.services.conversion_service import ConversionOptions, ConversionService
from PIL import Image


def _yolo_layout(root: Path, row: str, yaml_text: str = "") -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "images" / "sample.jpg").write_bytes(b"image")
    (root / "labels" / "sample.txt").write_text(row + "\n", encoding="utf-8")
    if yaml_text:
        (root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def test_segmentation_row_is_not_misdetected_as_pose(tmp_path):
    root = tmp_path / "segment"
    _yolo_layout(root, "0 0.1 0.1 0.3 0.1 0.4 0.3 0.3 0.5 0.1 0.3")

    detected = DatasetDetector.detect(root)

    assert detected.task_name == "yolo_segmentation"


def test_official_pose_uses_yaml_schema(tmp_path):
    root = tmp_path / "pose"
    _yolo_layout(
        root,
        "0 0.5 0.5 0.4 0.4 0.4 0.4 2 0.6 0.6 2",
        "task: pose\nkpt_shape: [2, 3]\nnames: [person]\n",
    )

    detected = DatasetDetector.detect(root)

    assert detected.task_name == "yolo_pose"


def test_scan_worker_reads_official_yaml_class_names(tmp_path):
    root = tmp_path / "detect"
    _yolo_layout(root, "1 0.5 0.5 0.4 0.4", "names:\n  0: person\n  1: vehicle\n")
    settings = ProjectSettings(
        image_dir=root / "images",
        annotation_dir=root / "labels",
        annotation_format="yolo",
        dataset_task="yolo_detection",
    )

    presets = DatasetScanWorker(
        root / "images", root / "labels", settings, root,
    )._discover_presets()

    assert [preset.name for preset in presets] == ["person", "vehicle"]


def test_conversion_uses_yaml_names_as_source_of_truth(tmp_path):
    root = tmp_path / "source"
    _yolo_layout(root, "1 0.5 0.5 0.4 0.4", "names: [person, vehicle]\n")
    Image.new("RGB", (100, 100)).save(root / "images" / "sample.jpg")
    output = tmp_path / "output"

    report = ConversionService().convert(ConversionOptions(
        source_format="yolo",
        source_path=root,
        output_format="voc",
        output_path=output,
        presets=[],
        source_task="yolo_detection",
        output_task="voc",
        overwrite=True,
    ))

    assert report.failed == 0 and report.succeeded == 1
    assert "<name>vehicle</name>" in (output / "Annotations" / "sample.xml").read_text(encoding="utf-8")
