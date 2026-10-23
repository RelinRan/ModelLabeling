"""Format and task names are technical identifiers with one canonical spelling.

Every dropdown, the status bar, the reports, and the error prompts read from
these helpers, so an alternative spelling ("VOC", "YOLO 检测") is a bug.
"""

from src.services.display_names import (
    format_name,
    format_options,
    format_task_name,
    task_label,
    task_options,
)


def test_format_names_are_canonical_in_every_language():
    assert [label for label, _ in format_options()] == ["COCO", "YOLO", "Pascal VOC"]
    assert [label for label, _ in format_options("en_US")] == ["COCO", "YOLO", "Pascal VOC"]
    assert format_name("voc") == "Pascal VOC"
    assert format_name("YOLO") == "YOLO"


def test_task_labels_pair_the_format_with_an_english_task_name():
    assert task_label("yolo_detection") == "YOLO Detection"
    assert task_label("yolo_segmentation") == "YOLO Segmentation"
    assert task_label("yolo_pose") == "YOLO Pose"
    assert task_label("yolo_obb") == "YOLO OBB"
    assert task_label("voc") == "Pascal VOC"
    assert task_label("coco") == "COCO"


def test_task_options_are_derived_from_the_format():
    assert [label for label, _ in task_options("yolo")] == [
        "YOLO Detection",
        "YOLO Segmentation",
        "YOLO Pose",
        "YOLO OBB",
    ]
    assert task_options("voc") == [("Pascal VOC", "voc")]
    assert task_options("coco") == [("COCO", "coco")]
    assert task_options("") == []


def test_status_label_separates_format_and_task_only_for_yolo():
    assert format_task_name("yolo", "yolo_segmentation") == "YOLO · Segmentation"
    assert format_task_name("voc", "voc") == "Pascal VOC"
    assert format_task_name("coco", "coco") == "COCO"
