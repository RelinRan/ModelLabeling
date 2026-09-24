# -*- coding: utf-8 -*-
"""The read-only validator: every defect the fixtures can inject is reported.

Two properties matter beyond "the rule fires":

* the scan writes nothing, so it is safe on a real dataset at any time;
* a file that merely sits beside the labels is never mistaken for an
  annotation -- the old orphan pass deleted those.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from src.models.annotation import LabelPreset
from src.services.cleanup import format_issue, scan_dataset
from src.services.cleanup.report import RULES, Severity


def rules_of(plan) -> set[str]:
    return {value.rule for value in plan.issues}


def presets_for(*names: str) -> list[LabelPreset]:
    return [LabelPreset(name, index, "#00e5ff") for index, name in enumerate(names)]


def tree_fingerprint(root) -> dict:
    """Every file below root with its size and mtime."""
    from pathlib import Path

    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in Path(root).rglob("*") if path.is_file()
    }


# --------------------------------------------------------------------------
# the scan must not write
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["voc", "yolo", "coco"])
def test_scan_writes_nothing(synthetic_dataset, fmt):
    """A scan is read-only -- including for COCO, whose store creates files
    the moment it is constructed."""
    root = synthetic_dataset(f"ro-{fmt}", fmt=fmt, frames=4)
    before = tree_fingerprint(root)
    plan = scan_dataset(root)
    assert plan.total_images == 4
    assert tree_fingerprint(root) == before


def test_scan_reports_a_clean_voc_dataset_as_clean(synthetic_dataset):
    """No defect injected, no finding raised: the baseline for idempotence."""
    root = synthetic_dataset("clean", fmt="voc", frames=6, classes=("obj",))
    plan = scan_dataset(root, presets=presets_for("obj"))
    assert plan.is_clean(), [format_issue(value) for value in plan.issues]


# --------------------------------------------------------------------------
# Pascal VOC
# --------------------------------------------------------------------------


@pytest.mark.parametrize("defect,expected", [
    ("no_object", "voc.annotation.missing"),
    ("small_box", "voc.box.small"),
    ("zero_area", "voc.box.degenerate"),
    ("duplicate_box", "voc.box.duplicate"),
    ("near_duplicate_box", "voc.box.near_duplicate"),
    ("label_conflict", "voc.box.geometry_conflict"),
    ("out_of_bounds", "voc.box.out_of_bounds"),
    ("size_mismatch", "voc.size.mismatch"),
    ("wrong_depth", "voc.depth.mismatch"),
    ("unreadable_image", "image.unreadable"),
    ("orphan_annotation", "voc.orphan"),
])
def test_voc_defects_are_reported(synthetic_dataset, defect, expected):
    root = synthetic_dataset(f"voc-{defect}", fmt="voc", frames=12,
                             classes=("4R", "4Ra"), defects=(defect,))
    plan = scan_dataset(root, presets=presets_for("4R", "4Ra"))
    assert expected in rules_of(plan), sorted(rules_of(plan))


def test_voc_unknown_label_is_reported(synthetic_dataset):
    """A box whose <name> is not a project label is a REVIEW finding."""
    root = synthetic_dataset("voc-unknown", fmt="voc", frames=3, classes=("obj",))
    plan = scan_dataset(root, presets=presets_for("something-else"))
    unknown = [value for value in plan.issues if value.rule == "voc.object.unknown_label"]
    assert len(unknown) == 3
    assert unknown[0].severity is Severity.REVIEW


def test_voc_blank_name_is_structural(synthetic_dataset):
    """_load_voc raises on a blank <name>, so the file is unusable as-is."""
    root = synthetic_dataset("voc-blank", fmt="voc", frames=2, classes=("obj",))
    for path in (root / "Annotations").glob("*.xml"):
        text = path.read_text(encoding="utf-8").replace("<name>obj</name>", "<name>  </name>")
        path.write_text(text, encoding="utf-8")
    plan = scan_dataset(root, presets=presets_for("obj"))
    blank = [value for value in plan.issues if value.rule == "voc.object.blank_name"]
    assert len(blank) == 2
    assert blank[0].severity is Severity.STRUCTURAL


# --------------------------------------------------------------------------
# YOLO
# --------------------------------------------------------------------------


@pytest.mark.parametrize("defect,expected", [
    ("no_object", "yolo.annotation.missing"),
    ("small_box", "yolo.row.small"),
    ("zero_area", "yolo.row.degenerate"),
    ("duplicate_box", "yolo.row.duplicate"),
    ("near_duplicate_box", "yolo.row.near_duplicate"),
    ("out_of_bounds", "yolo.row.out_of_range"),
    ("unreadable_image", "image.unreadable"),
    ("orphan_annotation", "yolo.orphan"),
])
def test_yolo_defects_are_reported(synthetic_dataset, defect, expected):
    root = synthetic_dataset(f"yolo-{defect}", fmt="yolo", frames=12, defects=(defect,))
    plan = scan_dataset(root, presets=presets_for("obj"))
    assert expected in rules_of(plan), sorted(rules_of(plan))


def test_yolo_unparseable_row_is_not_an_annotation(synthetic_dataset):
    """The old scan called any non-blank line an annotation (`any(line.strip())`),
    so a garbage row counted as a usable label."""
    root = synthetic_dataset("yolo-junk", fmt="yolo", frames=2)
    label = sorted((root / "labels" / "train").glob("*.txt"))[0]
    label.write_text("hello world\n", encoding="utf-8")
    plan = scan_dataset(root, presets=presets_for("obj"))
    # Two columns where five are required: the row is rejected, not counted.
    assert "yolo.row.bad_arity" in rules_of(plan)


def test_yolo_row_with_five_but_non_numeric_columns(synthetic_dataset):
    root = synthetic_dataset("yolo-nan", fmt="yolo", frames=2)
    label = sorted((root / "labels" / "train").glob("*.txt"))[0]
    label.write_text("0 a b c d\n", encoding="utf-8")
    plan = scan_dataset(root, presets=presets_for("obj"))
    assert "yolo.row.bad_number" in rules_of(plan)


def test_yolo_class_id_outside_the_table(synthetic_dataset):
    root = synthetic_dataset("yolo-class", fmt="yolo", frames=2)
    label = sorted((root / "labels" / "train").glob("*.txt"))[0]
    label.write_text("7 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    plan = scan_dataset(root, presets=presets_for("obj"))
    assert "yolo.row.class_unknown" in rules_of(plan)


def test_yolo_class_table_mismatch_is_review(synthetic_dataset):
    """classes.txt disagreeing with the project presets relabels the whole
    dataset without anything looking broken."""
    root = synthetic_dataset("yolo-presets", fmt="yolo", frames=2)
    (root / "classes.txt").write_text("a\nb\n", encoding="utf-8")
    plan = scan_dataset(root, presets=presets_for("x", "y"))
    mismatch = [value for value in plan.issues if value.rule == "yolo.classes.preset_mismatch"]
    assert len(mismatch) == 1
    assert mismatch[0].severity is Severity.REVIEW
    assert "a, b" in format_issue(mismatch[0])


def test_yolo_stale_cache_is_reported(synthetic_dataset):
    root = synthetic_dataset("yolo-cache", fmt="yolo", frames=2)
    (root / "labels" / "train" / "labels.cache").write_bytes(b"stale")
    plan = scan_dataset(root, presets=presets_for("obj"))
    assert "yolo.cache.present" in rules_of(plan)


# --------------------------------------------------------------------------
# COCO
# --------------------------------------------------------------------------


@pytest.mark.parametrize("defect,expected", [
    ("no_object", "coco.image_record.no_annotation"),
    ("small_box", "coco.annotation.small"),
    ("zero_area", "coco.annotation.bad_bbox"),
    ("duplicate_box", "coco.annotation.duplicate"),
    ("near_duplicate_box", "coco.annotation.near_duplicate"),
    ("label_conflict", "coco.annotation.geometry_conflict"),
    ("out_of_bounds", "coco.annotation.out_of_bounds"),
])
def test_coco_defects_are_reported(synthetic_dataset, defect, expected):
    root = synthetic_dataset(f"coco-{defect}", fmt="coco", frames=12,
                             classes=("4R", "4Ra"), defects=(defect,))
    plan = scan_dataset(root, presets=presets_for("4R", "4Ra"))
    assert expected in rules_of(plan), sorted(rules_of(plan))


def test_coco_dangling_reference_and_unknown_category(synthetic_dataset):
    """Referential integrity: pycocotools mis-evaluates without raising."""
    import json

    root = synthetic_dataset("coco-refs", fmt="coco", frames=3)
    path = root / "annotations.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["annotations"][0]["image_id"] = 999
    document["annotations"][1]["category_id"] = 42
    path.write_text(json.dumps(document), encoding="utf-8")

    plan = scan_dataset(root, presets=presets_for("obj"))
    found = rules_of(plan)
    assert "coco.annotation.dangling_image" in found
    assert "coco.annotation.unknown_category" in found
    # The image left without its annotation is reported as unannotated.
    assert "coco.image_record.no_annotation" in found


def test_coco_record_without_a_file_on_disk(synthetic_dataset):
    """The old cleanup only pruned records for images *it* had deleted, so a
    record whose file vanished outside the app stayed forever."""
    import json

    root = synthetic_dataset("coco-ghost", fmt="coco", frames=3)
    path = root / "annotations.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["images"][0]["file_name"] = "does-not-exist.jpg"
    path.write_text(json.dumps(document), encoding="utf-8")

    plan = scan_dataset(root, presets=presets_for("obj"))
    missing = [value for value in plan.issues if value.rule == "coco.image_record.missing_file"]
    assert len(missing) == 1
    assert missing[0].severity is Severity.STRUCTURAL


def test_coco_duplicate_and_dangling_ids(synthetic_dataset):
    import json

    root = synthetic_dataset("coco-ids", fmt="coco", frames=3)
    path = root / "annotations.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["images"][1]["id"] = document["images"][0]["id"]
    path.write_text(json.dumps(document), encoding="utf-8")

    plan = scan_dataset(root, presets=presets_for("obj"))
    assert "coco.image.duplicate_id" in rules_of(plan)


def test_coco_area_mismatch_is_structural(synthetic_dataset):
    import json

    root = synthetic_dataset("coco-area", fmt="coco", frames=2)
    path = root / "annotations.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["annotations"][0]["area"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")

    plan = scan_dataset(root, presets=presets_for("obj"))
    area = [value for value in plan.issues if value.rule == "coco.annotation.area_mismatch"]
    assert area and area[0].severity is Severity.STRUCTURAL


# --------------------------------------------------------------------------
# the orphan pass must not delete a file that is not an annotation
# --------------------------------------------------------------------------


def test_notes_beside_the_labels_is_not_an_orphan(synthetic_dataset):
    """The old pass deleted every *.txt in the label directory whose stem had
    no image, sparing only classes.txt -- so notes.txt was data loss."""
    root = synthetic_dataset("yolo-notes", fmt="yolo", frames=2)
    notes = root / "labels" / "train" / "notes.txt"
    notes.write_text("frame 40-60 skipped, subject left the frame\n", encoding="utf-8")

    plan = scan_dataset(root, presets=presets_for("obj"))
    found = rules_of(plan)
    assert "yolo.orphan" not in found
    assert "yolo.unrecognized" in found
    assert notes.is_file()


def test_a_real_orphan_is_still_an_orphan(synthetic_dataset):
    root = synthetic_dataset("yolo-real-orphan", fmt="yolo", frames=2)
    orphan = root / "labels" / "train" / "GONE_000001.txt"
    orphan.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    plan = scan_dataset(root, presets=presets_for("obj"))
    assert "yolo.orphan" in rules_of(plan)


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def test_two_images_sharing_a_stem_are_reported(synthetic_dataset):
    """foo.jpg + foo.png mirror onto one foo.xml, so they overwrite each
    other's boxes. The old scan looked at one image at a time and missed it."""
    from PIL import Image

    root = synthetic_dataset("stems", fmt="voc", frames=2)
    Image.new("RGB", (640, 480), (1, 2, 3)).save(root / "JPEGImages" / "shared.png")
    Image.new("RGB", (640, 480), (4, 5, 6)).save(root / "JPEGImages" / "shared.jpg")

    plan = scan_dataset(root, presets=presets_for("obj"))
    conflicts = [value for value in plan.issues if value.rule == "image.stem_conflict"]
    assert len(conflicts) == 2
    assert conflicts[0].severity is Severity.STRUCTURAL


# --------------------------------------------------------------------------
# the pixel probe
# --------------------------------------------------------------------------


def test_truncated_jpeg_needs_a_pixel_decode_to_find(synthetic_dataset):
    """A truncated JPEG keeps a valid header: its size reads fine and only a
    decode fails. Without the probe it reaches the save path and breaks it."""
    root = synthetic_dataset("truncated", fmt="voc", frames=4, defects=("unreadable_image",))

    fast = scan_dataset(root, presets=presets_for("obj"), verify_pixels=False)
    deep = scan_dataset(root, presets=presets_for("obj"), verify_pixels=True)

    assert "image.unreadable" not in rules_of(fast)
    assert "image.unreadable" in rules_of(deep)
    assert deep.total_images == fast.total_images == 4


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def test_every_rule_has_both_languages():
    for rule, spec in RULES.items():
        assert spec.tag[0] and spec.tag[1], rule
        assert spec.template[0] and spec.template[1], rule
        assert isinstance(spec.severity, Severity), rule


def test_format_issue_renders_the_finding(synthetic_dataset):
    root = synthetic_dataset("render", fmt="voc", frames=2, defects=("small_box",))
    plan = scan_dataset(root, presets=presets_for("obj"))
    small = next(value for value in plan.issues if value.rule == "voc.box.small")

    zh = format_issue(small)
    en = format_issue(small, english=True)
    assert zh.startswith("[过小框] ")
    assert en.startswith("[Tiny box] ")
    assert "12" in zh and "20" in zh


def test_unknown_rule_id_is_rejected_at_construction():
    from pathlib import Path

    from src.services.cleanup.report import issue

    with pytest.raises(KeyError):
        issue("not.a.rule", Path("x"))
